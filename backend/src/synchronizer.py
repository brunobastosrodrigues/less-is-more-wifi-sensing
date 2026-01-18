#!/usr/bin/env python3
"""
Frame Synchronizer for WiFi CSI TDMA System

Converts UDP streams from ESP32 nodes into synchronized numpy tensors.

Firmware Compatibility:
- v1 protocol: 128-byte packets (magic 0xC5 0x02) - legacy
- v2 protocol: 148-byte packets (magic 0xC5 0x03)
- v5 protocol: batch packets (magic 0xC5 0x07)

Architecture:
- Binary packet parser with auto-detection
- Per-link ring buffers with timestamps
- Frame builder outputs (N_links, 52) tensors
- CRC32 validation for data integrity
"""

import struct
import asyncio
import logging
import time
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Callable
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# Binary CSI packet constants
BINARY_CSI_MAGIC_V1 = bytes([0xC5, 0x02])
BINARY_CSI_PACKET_SIZE_V1 = 128

BINARY_CSI_MAGIC_V2 = bytes([0xC5, 0x03])
BINARY_CSI_PACKET_SIZE_V2 = 148

BINARY_CSI_MAGIC_V5 = bytes([0xC5, 0x07])
BINARY_BATCH_HEADER_SIZE = 8
BINARY_SAMPLE_SIZE_V5 = 107

BINARY_CSI_NUM_CARRIERS = 52

# Frame builder settings
DEFAULT_FRAME_RATE = 20  # Hz
DEFAULT_BUFFER_SIZE = 10  # Packets per link
FRAME_WINDOW_MS = 50  # Collect packets within this window


def _crc32(data: bytes) -> int:
    """Compute CRC32 (IEEE polynomial)."""
    return zlib.crc32(data) & 0xFFFFFFFF


@dataclass
class BinaryCSIPacket:
    """Parsed binary CSI packet from ESP32."""
    tx_mac: str
    rx_mac: str
    seq: int
    timestamp_us: int
    rssi: int
    noise_floor: int
    channel: int
    bandwidth: int
    num_carriers: int
    iq_data: np.ndarray  # Shape: (52, 2) - I/Q pairs
    crc32: int
    received_at: float = field(default_factory=time.time)
    sample_idx: int = 0
    sample_count: int = 1

    @property
    def amplitude(self) -> np.ndarray:
        """Compute amplitude from I/Q."""
        return np.sqrt(self.iq_data[:, 0]**2 + self.iq_data[:, 1]**2)

    @property
    def phase(self) -> np.ndarray:
        """Compute phase from I/Q."""
        return np.arctan2(self.iq_data[:, 1], self.iq_data[:, 0])

    def to_dict(self) -> dict:
        """Serialize for JSON/logging."""
        return {
            'tx_mac': self.tx_mac,
            'rx_mac': self.rx_mac,
            'seq': self.seq,
            'timestamp_us': self.timestamp_us,
            'rssi': self.rssi,
            'amplitude': self.amplitude.tolist(),
            'received_at': self.received_at,
        }


def parse_binary_csi_packet_v1(data: bytes, rx_mac: str) -> Optional[BinaryCSIPacket]:
    """Parse 128-byte binary CSI packet (v1 format)."""
    if len(data) != BINARY_CSI_PACKET_SIZE_V1:
        return None
    if data[0:2] != BINARY_CSI_MAGIC_V1:
        return None

    try:
        tx_mac = ':'.join(f'{b:02X}' for b in data[2:8])
        seq, timestamp_us = struct.unpack('<HI', data[8:14])
        rssi = struct.unpack('<b', data[14:15])[0]
        noise_floor = struct.unpack('<b', data[15:16])[0]
        channel = data[16]
        bandwidth = data[17]
        num_carriers = data[18]

        iq_raw = struct.unpack('<104b', data[20:124])
        iq_data = np.array(iq_raw, dtype=np.float32).reshape(52, 2)
        crc32 = struct.unpack('<I', data[124:128])[0]

        return BinaryCSIPacket(
            tx_mac=tx_mac, rx_mac=rx_mac, seq=seq, timestamp_us=timestamp_us,
            rssi=rssi, noise_floor=noise_floor, channel=channel,
            bandwidth=bandwidth, num_carriers=num_carriers,
            iq_data=iq_data, crc32=crc32
        )
    except Exception as e:
        logger.error(f"Failed to parse v1 CSI packet: {e}")
        return None


def parse_binary_csi_packet_v2(data: bytes, rx_mac_from_ip: str) -> Optional[BinaryCSIPacket]:
    """Parse 148-byte binary CSI packet (v2 format)."""
    if len(data) != BINARY_CSI_PACKET_SIZE_V2:
        return None
    if data[0:2] != BINARY_CSI_MAGIC_V2:
        return None

    try:
        tx_mac = ':'.join(f'{b:02X}' for b in data[2:8])
        rx_mac = ':'.join(f'{b:02X}' for b in data[8:14])
        if rx_mac == '00:00:00:00:00:00':
            rx_mac = rx_mac_from_ip

        seq = struct.unpack('<H', data[14:16])[0]
        sample_idx = data[16]
        sample_count = data[17]
        timestamp_us = struct.unpack('<I', data[18:22])[0]
        rssi = struct.unpack('<b', data[22:23])[0]
        noise_floor = struct.unpack('<b', data[23:24])[0]
        channel = data[25]
        bandwidth = data[26]
        num_carriers = data[28]

        iq_raw = struct.unpack('<104b', data[40:144])
        iq_data = np.array(iq_raw, dtype=np.float32).reshape(52, 2)
        crc32 = struct.unpack('<I', data[144:148])[0]

        return BinaryCSIPacket(
            tx_mac=tx_mac, rx_mac=rx_mac, seq=seq, timestamp_us=timestamp_us,
            rssi=rssi, noise_floor=noise_floor, channel=channel,
            bandwidth=bandwidth, num_carriers=num_carriers,
            iq_data=iq_data, crc32=crc32,
            sample_idx=sample_idx, sample_count=sample_count
        )
    except Exception as e:
        logger.error(f"Failed to parse v2 CSI packet: {e}")
        return None


def parse_binary_csi_batch_v5(
    data: bytes,
    rx_mac: str,
    slot_resolver: Callable[[int], Optional[str]]
) -> List[BinaryCSIPacket]:
    """Parse v5 batch CSI packet containing multiple samples."""
    min_size = BINARY_BATCH_HEADER_SIZE + BINARY_SAMPLE_SIZE_V5 + 4
    if len(data) < min_size:
        return []
    if data[0:2] != BINARY_CSI_MAGIC_V5:
        return []

    # Validate CRC32
    received_crc = struct.unpack('<I', data[-4:])[0]
    computed_crc = _crc32(data[:-4])
    if received_crc != computed_crc:
        logger.warning(f"v5 batch CRC mismatch")
        return []

    try:
        slot_id = data[2]
        sample_count = data[3]
        seq = struct.unpack('<H', data[4:6])[0]
        timestamp_lo = struct.unpack('<H', data[6:8])[0]

        tx_mac = slot_resolver(slot_id)
        if not tx_mac:
            logger.warning(f"Unknown slot_id: {slot_id}")
            return []

        packets = []
        offset = BINARY_BATCH_HEADER_SIZE

        for i in range(sample_count):
            if offset + BINARY_SAMPLE_SIZE_V5 > len(data) - 4:
                break

            timestamp_delta = struct.unpack('<H', data[offset:offset+2])[0]
            rssi = struct.unpack('<b', data[offset+2:offset+3])[0]
            iq_raw = struct.unpack('<104b', data[offset+3:offset+107])
            iq_data = np.array(iq_raw, dtype=np.float32).reshape(52, 2)
            timestamp_us = (timestamp_lo + timestamp_delta) & 0xFFFF

            packet = BinaryCSIPacket(
                tx_mac=tx_mac, rx_mac=rx_mac, seq=seq,
                timestamp_us=timestamp_us, rssi=rssi,
                noise_floor=-90, channel=0, bandwidth=20,
                num_carriers=52, iq_data=iq_data, crc32=received_crc,
                sample_idx=i + 1, sample_count=sample_count,
            )
            packets.append(packet)
            offset += BINARY_SAMPLE_SIZE_V5

        return packets

    except Exception as e:
        logger.error(f"Failed to parse v5 batch: {e}")
        return []


@dataclass
class LinkBuffer:
    """Ring buffer for a single TX->RX link."""
    tx_mac: str
    rx_mac: str
    packets: deque = field(default_factory=lambda: deque(maxlen=DEFAULT_BUFFER_SIZE))
    last_rssi: int = -90
    last_amplitude: Optional[np.ndarray] = None
    packet_count: int = 0

    def add(self, packet: BinaryCSIPacket):
        """Add packet to buffer."""
        self.packets.append(packet)
        self.last_rssi = packet.rssi
        self.last_amplitude = packet.amplitude
        self.packet_count += 1

    def get_latest(self, max_age_ms: float = FRAME_WINDOW_MS) -> Optional[BinaryCSIPacket]:
        """Get most recent packet within age limit."""
        if not self.packets:
            return None

        now = time.time()
        for packet in reversed(self.packets):
            age_ms = (now - packet.received_at) * 1000
            if age_ms <= max_age_ms:
                return packet
        return None


class FrameSynchronizer:
    """
    Synchronizes CSI packets from multiple ESP32 nodes into coherent frames.

    Converts chaotic UDP arrival into structured (N_links, 52) numpy tensors.
    """

    def __init__(self, frame_rate: float = DEFAULT_FRAME_RATE, scheduler=None):
        self.frame_rate = frame_rate
        self.frame_interval = 1.0 / frame_rate
        self._scheduler = scheduler

        # Node tracking: MAC -> IP
        self.nodes: Dict[str, str] = {}
        self.ip_to_mac: Dict[str, str] = {}

        # Link buffers: (tx_mac, rx_mac) -> LinkBuffer
        self.links: Dict[Tuple[str, str], LinkBuffer] = {}
        self.link_order: List[Tuple[str, str]] = []
        self._frame_callbacks: List[Callable] = []

        # Statistics
        self.packets_received = 0
        self.packets_parsed = 0
        self.frames_built = 0
        self.last_frame_time = time.time()

        # Frame building task
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def register_node(self, mac: str, ip: str):
        """Register a node's MAC-IP mapping."""
        mac = mac.upper()
        self.nodes[mac] = ip
        self.ip_to_mac[ip] = mac
        logger.debug(f"Registered node {mac[-5:]} at {ip}")
        self._rebuild_link_order()

    def _rebuild_link_order(self):
        """Rebuild consistent link ordering for tensor output."""
        nodes = sorted(self.nodes.keys())
        self.link_order = []
        for tx in nodes:
            for rx in nodes:
                if tx != rx:
                    self.link_order.append((tx, rx))

    def _get_or_create_link(self, tx_mac: str, rx_mac: str) -> LinkBuffer:
        """Get or create link buffer."""
        key = (tx_mac, rx_mac)
        if key not in self.links:
            self.links[key] = LinkBuffer(tx_mac=tx_mac, rx_mac=rx_mac)
        return self.links[key]

    def _resolve_slot_id(self, slot_id: int) -> Optional[str]:
        """Resolve slot_id to TX MAC address."""
        if self._scheduler:
            return self._scheduler.get_tx_mac_for_slot(slot_id)

        nodes = sorted(self.nodes.keys())
        if 0 <= slot_id < len(nodes):
            return nodes[slot_id]
        return None

    def process_packet(self, data: bytes, source_ip: str) -> Optional[BinaryCSIPacket]:
        """Process incoming UDP packet."""
        self.packets_received += 1

        rx_mac = self.ip_to_mac.get(source_ip)
        if not rx_mac:
            return None

        # v5 batch packets
        if len(data) >= BINARY_BATCH_HEADER_SIZE and data[0:2] == BINARY_CSI_MAGIC_V5:
            packets = parse_binary_csi_batch_v5(data, rx_mac, self._resolve_slot_id)
            for packet in packets:
                self.packets_parsed += 1
                link = self._get_or_create_link(packet.tx_mac, packet.rx_mac)
                link.add(packet)
            return packets if packets else None

        # v2 packets
        if len(data) == BINARY_CSI_PACKET_SIZE_V2 and data[0:2] == BINARY_CSI_MAGIC_V2:
            packet = parse_binary_csi_packet_v2(data, rx_mac)
            if packet:
                self.packets_parsed += 1
                link = self._get_or_create_link(packet.tx_mac, packet.rx_mac)
                link.add(packet)
                return packet

        # v1 packets
        if len(data) == BINARY_CSI_PACKET_SIZE_V1 and data[0:2] == BINARY_CSI_MAGIC_V1:
            packet = parse_binary_csi_packet_v1(data, rx_mac)
            if packet:
                self.packets_parsed += 1
                link = self._get_or_create_link(packet.tx_mac, packet.rx_mac)
                link.add(packet)
                return packet

        return None

    def build_frame(self) -> Optional[Tuple[np.ndarray, Dict]]:
        """Build synchronized frame from current link buffers."""
        if not self.link_order:
            return None

        n_links = len(self.link_order)
        n_carriers = BINARY_CSI_NUM_CARRIERS

        tensor = np.zeros((n_links, n_carriers), dtype=np.float32)
        rssi = np.zeros(n_links, dtype=np.float32)
        valid_links = 0
        link_info = []

        for i, (tx, rx) in enumerate(self.link_order):
            link = self.links.get((tx, rx))
            if link:
                packet = link.get_latest(max_age_ms=FRAME_WINDOW_MS * 2)
                if packet:
                    tensor[i, :] = packet.amplitude
                    rssi[i] = packet.rssi
                    valid_links += 1
                    link_info.append({
                        'tx': tx[-5:], 'rx': rx[-5:],
                        'rssi': packet.rssi, 'seq': packet.seq
                    })
                elif link.last_amplitude is not None:
                    tensor[i, :] = link.last_amplitude * 0.9
                    rssi[i] = link.last_rssi

        self.frames_built += 1
        self.last_frame_time = time.time()

        metadata = {
            'timestamp': time.time(),
            'n_links': n_links,
            'valid_links': valid_links,
            'frame_id': self.frames_built,
            'rssi': rssi.tolist(),
            'links': link_info,
        }

        return tensor, metadata

    def on_frame(self, callback: Callable):
        """Register callback for new frames."""
        self._frame_callbacks.append(callback)

    async def _frame_loop(self):
        """Background task that builds frames at fixed rate."""
        logger.info(f"Frame builder started at {self.frame_rate}Hz")

        while self._running:
            start = time.time()
            result = self.build_frame()

            if result:
                tensor, metadata = result
                for callback in self._frame_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(tensor, metadata)
                        else:
                            callback(tensor, metadata)
                    except Exception as e:
                        logger.error(f"Frame callback error: {e}")

            elapsed = time.time() - start
            sleep_time = max(0, self.frame_interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def start(self):
        """Start the frame builder."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._frame_loop())

    async def stop(self):
        """Stop the frame builder."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    def get_stats(self) -> dict:
        """Get synchronizer statistics."""
        return {
            'nodes': len(self.nodes),
            'links_expected': len(self.link_order),
            'links_active': len(self.links),
            'packets_received': self.packets_received,
            'packets_parsed': self.packets_parsed,
            'frames_built': self.frames_built,
            'frame_rate': self.frame_rate,
        }
