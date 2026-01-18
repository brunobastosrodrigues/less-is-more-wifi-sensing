#!/usr/bin/env python3
"""
Build Training Dataset from Raw CSI Data

Processes raw CSI JSONL files to create a training-ready dataset:
1. Streams raw CSI packets
2. Windows them (configurable window size)
3. Extracts per-link features
4. Outputs Parquet or CSV for ML training

Usage:
    python build_dataset.py \
        --input-dir /path/to/csi_data \
        --output /path/to/dataset.parquet \
        --window-ms 250 \
        --dates 2026-01-01 2026-01-02

Requirements:
    - numpy, pandas
    - pyarrow (optional, for Parquet output)
"""

import argparse
import gzip
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Generator, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CSIPacket:
    """Single CSI packet."""
    timestamp: float
    tx_mac: str
    rx_mac: str
    rssi: int
    csi_data: List[int]
    seq: int = 0


@dataclass
class CSIWindow:
    """A window of CSI packets."""
    start_time: float
    end_time: float
    packets: List[CSIPacket] = field(default_factory=list)

    def get_link_features(self) -> Dict[str, Dict]:
        """Extract features per link."""
        links = defaultdict(list)
        for pkt in self.packets:
            link_id = f"{pkt.tx_mac}->{pkt.rx_mac}"
            links[link_id].append({
                'rssi': pkt.rssi,
                'csi': pkt.csi_data,
                'timestamp': pkt.timestamp
            })

        features = {}
        for link_id, samples in links.items():
            rssi_vals = [s['rssi'] for s in samples]

            # Extract amplitude from I/Q pairs
            amplitudes = []
            for s in samples:
                csi = s['csi']
                amp = []
                for i in range(0, len(csi), 2):
                    if i + 1 < len(csi):
                        I, Q = csi[i], csi[i + 1]
                        amp.append(np.sqrt(I**2 + Q**2))
                if amp:
                    amplitudes.append(amp)

            # Compute statistical features
            if amplitudes:
                amp_array = np.array(amplitudes)
                amp_mean = np.mean(amp_array, axis=0)
                amp_std = np.std(amp_array, axis=0) if len(amplitudes) > 1 else np.zeros_like(amp_mean)
            else:
                amp_mean = []
                amp_std = []

            features[link_id] = {
                'rssi_mean': float(np.mean(rssi_vals)) if rssi_vals else -100,
                'rssi_std': float(np.std(rssi_vals)) if len(rssi_vals) > 1 else 0,
                'rssi_min': float(np.min(rssi_vals)) if rssi_vals else -100,
                'rssi_max': float(np.max(rssi_vals)) if rssi_vals else -100,
                'n_packets': len(samples),
                'amplitude_mean': amp_mean.tolist() if len(amp_mean) else [],
                'amplitude_std': amp_std.tolist() if len(amp_std) else [],
                'amplitude_var': float(np.var(amp_mean)) if len(amp_mean) else 0,
            }

        return features


def parse_timestamp(ts_str: str) -> float:
    """Convert ISO timestamp to Unix time."""
    try:
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        return dt.timestamp()
    except:
        return 0.0


def stream_csi_packets(file_path: Path) -> Generator[CSIPacket, None, None]:
    """Stream CSI packets from a JSONL file."""
    opener = gzip.open if file_path.suffix == '.gz' else open

    try:
        with opener(file_path, 'rt') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    ts = data.get('received_at', data.get('timestamp', ''))

                    yield CSIPacket(
                        timestamp=parse_timestamp(ts) if isinstance(ts, str) else float(ts),
                        tx_mac=data.get('tx_mac', ''),
                        rx_mac=data.get('rx_mac', ''),
                        rssi=data.get('rssi', -100),
                        csi_data=data.get('csi_data', []),
                        seq=data.get('seq', 0)
                    )
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")


def window_packets(
    packets: Generator[CSIPacket, None, None],
    window_ms: float = 250
) -> Generator[CSIWindow, None, None]:
    """Group packets into fixed-size time windows."""
    window_sec = window_ms / 1000.0
    current_window = None

    for pkt in packets:
        if pkt.timestamp == 0:
            continue

        if current_window is None:
            current_window = CSIWindow(
                start_time=pkt.timestamp,
                end_time=pkt.timestamp + window_sec
            )

        if pkt.timestamp < current_window.end_time:
            current_window.packets.append(pkt)
        else:
            # Yield current window and start new one
            if current_window.packets:
                yield current_window

            current_window = CSIWindow(
                start_time=pkt.timestamp,
                end_time=pkt.timestamp + window_sec,
                packets=[pkt]
            )

    # Yield final window
    if current_window and current_window.packets:
        yield current_window


def process_day(
    data_dir: Path,
    date_str: str,
    window_ms: float = 250
) -> List[Dict]:
    """Process all CSI files for a single day."""
    day_dir = data_dir / date_str
    if not day_dir.exists():
        logger.warning(f"Directory not found: {day_dir}")
        return []

    records = []
    files = sorted(day_dir.glob("csi_*.jsonl*"))
    logger.info(f"Processing {len(files)} files for {date_str}")

    for csi_file in files:
        packets = stream_csi_packets(csi_file)
        windows = window_packets(packets, window_ms)

        for window in windows:
            features = window.get_link_features()

            if not features:
                continue

            # Create a record per window
            record = {
                'date': date_str,
                'timestamp': window.start_time,
                'window_start': datetime.fromtimestamp(window.start_time).isoformat(),
                'window_end': datetime.fromtimestamp(window.end_time).isoformat(),
                'n_links': len(features),
                'n_packets': sum(f['n_packets'] for f in features.values()),
            }

            # Add per-link features (flattened)
            for link_id, link_features in features.items():
                prefix = link_id.replace(':', '').replace('->', '_')
                record[f'{prefix}_rssi_mean'] = link_features['rssi_mean']
                record[f'{prefix}_rssi_std'] = link_features['rssi_std']
                record[f'{prefix}_n_packets'] = link_features['n_packets']
                record[f'{prefix}_amp_var'] = link_features['amplitude_var']

            records.append(record)

    return records


def main():
    parser = argparse.ArgumentParser(description='Build dataset from raw CSI')
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Directory containing CSI data (date subdirs)')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output file (Parquet or CSV)')
    parser.add_argument('--dates', nargs='+', default=None,
                        help='Specific dates to process (YYYY-MM-DD)')
    parser.add_argument('--window-ms', type=float, default=250,
                        help='Window size in milliseconds')
    parser.add_argument('--format', choices=['parquet', 'csv'], default='parquet',
                        help='Output format')
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Building CSI Dataset")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Window: {args.window_ms}ms")

    # Determine dates to process
    if args.dates:
        dates = args.dates
    else:
        # Auto-detect from directory structure
        dates = sorted([
            d.name for d in args.input_dir.iterdir()
            if d.is_dir() and d.name.startswith('20')
        ])

    logger.info(f"Processing {len(dates)} days: {dates[0]} to {dates[-1]}")

    # Process each day
    all_records = []
    for date_str in dates:
        records = process_day(args.input_dir, date_str, args.window_ms)
        all_records.extend(records)
        logger.info(f"  {date_str}: {len(records)} windows")

    logger.info(f"Total: {len(all_records)} windows")

    # Create DataFrame
    df = pd.DataFrame(all_records)

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.format == 'parquet':
        try:
            df.to_parquet(args.output, compression='zstd', index=False)
        except ImportError:
            logger.warning("PyArrow not available, falling back to CSV")
            args.output = args.output.with_suffix('.csv')
            df.to_csv(args.output, index=False)
    else:
        df.to_csv(args.output, index=False)

    logger.info(f"Saved to {args.output}")
    logger.info(f"Columns: {len(df.columns)}")
    logger.info(f"Shape: {df.shape}")


if __name__ == '__main__':
    main()
