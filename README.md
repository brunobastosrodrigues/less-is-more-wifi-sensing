# Less is More - ESP32-C3 WiFi CSI Sensing

We empirically show the 'Less is More' dilution effect, showing that adding nodes introduces tradeoffs where bandwidth saturation and TDMA limitations force sampling rates below the Nyquist threshold. Strategic and sparse sensor placement is superior to dense meshes/complex algorithimic solutions.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a0301509-113c-4c70-9528-9eec0db3c313" width="400">
</p>

## Overview

This project provides a complete system for distributed WiFi sensing using CSI data. It implements:

- **Ring-based TDMA Protocol**: Collision-free transmission scheduling for multi-node WiFi sensing
- **Binary Protocol**: Zero-malloc, CRC32-validated communication between nodes and server
- **Real-time Synchronization**: Converts chaotic UDP streams into coherent CSI tensors

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    TDMA Cycle (N nodes)                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Backend (Python)        ESP32 Nodes                           │
│  ────────────────        ────────────                          │
│                                                                │
│  TRIGGER(Node1) ───────▶ Node1: TX burst packets               │
│                          Others: Capture CSI → Ring buffer     │
│                                                                │
│  TRIGGER(Node2) ───────▶ Node2: TX burst packets               │
│                          Others: Send CSI, capture new         │
│                                                                │
│  ... (repeats for all nodes) ...                               │
│                                                                │
│  One cycle complete ──▶  Effective rate: ~5-10 Hz              │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

## Features

- **Scalable**: Supports 2-16 ESP32-C3 nodes
- **Configurable Timing**: Adjust slot duration, burst count, and cycle delay
- **Reliable**: Watchdog timers, heap monitoring, auto-reconnect
- **Efficient**: Binary protocol with CRC32 validation, IRAM-optimized callbacks
- **Synchronized**: Frame builder produces (N_links, 52) numpy tensors
- **Analysis Scripts**: Complete evaluation pipeline for paper reproducibility

## Hardware Requirements

- **ESP32-C3 boards**: Tested with Seeed XIAO ESP32-C3
- **WiFi Access Point**: All nodes must connect to the same AP on the same channel
- **Server**: Any machine running Python 3.8+ or Docker (Linux recommended)

## Quick Start

### Option A: Docker (Recommended)

Docker provides the easiest way to run the backend with guaranteed reproducibility.

```bash
# 1. Clone and enter the directory
cd opensource-release

# 2. Create data directory
mkdir -p data/csi

# 3. Build and run
docker compose up --build
```

**Important**: The container uses `network_mode: host` which is required for UDP communication with ESP32 nodes. This means the container shares the host's network stack.

To stop:
```bash
docker compose down
```

### Option B: Native Python

#### 1. Flash the Firmware

```bash
cd firmware/esp-idf

# IMPORTANT: Configure WiFi credentials in sdkconfig.defaults:
# CONFIG_WIFI_SSID="your_wifi_ssid"
# CONFIG_WIFI_PASSWORD="your_wifi_password"
# CONFIG_SERVER_IP="192.168.1.100"  # Your backend server IP

# Clean and build (MUST delete sdkconfig first)
rm -f sdkconfig
idf.py set-target esp32c3
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

Repeat for each ESP32-C3 node.

#### 2. Start the Backend

```bash
cd backend/src

# Install dependencies
pip install numpy

# Run the server
python main.py
```

The server will:
1. Listen for ESP32 node registrations on UDP port 5000
2. Wait for at least 2 nodes to come online
3. Start TDMA scheduling, broadcasting triggers on UDP port 5001
4. Collect and synchronize CSI data from all nodes

#### 3. Verify Operation

You should see output like:
```
Server ready on UDP port 5000
Waiting for 2+ ESP32 nodes to come online...
HELLO from 1C:D5 (ESP32-C3 v5.1.0-PURE) at 192.168.1.101
HELLO from 2A:F3 (ESP32-C3 v5.1.0-PURE) at 192.168.1.102
TDMA Scheduler started
Status: nodes=2/2, cycles=10, triggers=40, packets=320, frames=100
```

## Configuration

### Backend (config.py)

```python
SLOT_DURATION_MS = 20        # Time per node slot (ms)
BURSTS_PER_TRIGGER = 10      # Packets per burst
CYCLE_DELAY_MS = 5           # Guard time between slots (ms)
MIN_NODES_TO_START = 2       # Minimum nodes to start scheduling
```

### Firmware (main.c)

```c
#define CONFIG_WIFI_SSID "your_wifi_ssid"
#define CONFIG_WIFI_PASSWORD "your_wifi_password"
#define CONFIG_SERVER_IP "192.168.1.100"
#define CONFIG_WIFI_CHANNEL 11  // Lock to specific channel
```

## Sampling Rate

The effective CSI sampling rate depends on the number of nodes and timing configuration:

```
Rate (Hz) = 1000 / (N × (slot_ms + cycle_delay_ms))
```

| Nodes | Slot (ms) | Delay (ms) | Rate (Hz) |
|-------|-----------|------------|-----------|
| 9 | 20 | 5 | 4.4 |
| 7 | 20 | 5 | 5.7 |
| 4 | 20 | 5 | 10.0 |
| 3 | 20 | 5 | 13.3 |

## Directory Structure

```
opensource-release/
├── docker-compose.yml       # Docker orchestration
├── firmware/
│   └── esp-idf/
│       └── main/
│           └── main.c       # ESP32-C3 firmware
├── backend/
│   ├── Dockerfile           # Container build
│   ├── requirements.txt     # Python dependencies
│   └── src/
│       ├── main.py          # Entry point
│       ├── config.py        # Configuration
│       ├── models.py        # Data models
│       ├── registry.py      # Node registry
│       ├── receiver.py      # UDP receiver
│       ├── synchronizer.py  # Frame synchronizer
│       └── scheduler/
│           └── tdma_scheduler.py
├── analysis/                # Paper reproducibility scripts
│   ├── evaluation/          # Presence detection evaluation
│   ├── preprocessing/       # Data preparation
│   ├── visualization/       # Publication figures
│   ├── utils/               # Statistical tests, reproducibility
│   └── REPRODUCIBILITY.md   # Step-by-step guide
├── docs/
│   └── TDMA_TIMING_DESIGN.md
├── LICENSE
└── README.md
```

## Protocol Details

### Binary Packet Formats

| Packet | Size | Magic | Purpose |
|--------|------|-------|---------|
| HELLO | 40 bytes | 0xC5 0x01 | Node registration |
| HEARTBEAT | 24 bytes | 0xC5 0x02 | Keepalive |
| TRIGGER | 32 bytes | 0xC5 0x10 | TX command |
| CSI Batch | Variable | 0xC5 0x07 | CSI data (v5) |

All packets include CRC32 validation.

### UDP Ports

| Port | Direction | Purpose |
|------|-----------|---------|
| 5000 | Node → Server | CSI data, HELLO, HEARTBEAT |
| 5001 | Server → Nodes | TRIGGER commands (broadcast) |
| 5555 | Node → Nodes | Burst packets (broadcast) |

## Use Cases

This system is designed for:

- **Presence Detection**: Binary occupied/empty classification
- **Zone Occupancy**: Room quadrant localization
- **Motion State**: Moving vs. stationary detection
- **Breathing Detection**: Respiratory rate monitoring (stationary subjects)

**Not suitable for** (requires >20 Hz sampling):
- Gesture recognition
- Centimeter-level tracking
- Multi-person tracking in same zone

## Reproducing Paper Results

See `analysis/REPRODUCIBILITY.md` for detailed instructions. Quick start:

```bash
cd analysis

# Install dependencies
pip install -r requirements.txt

# Run evaluation
python evaluation/evaluate_presence_detection.py \
    --data-dir /path/to/csi_data \
    --labels /path/to/labels.csv \
    --output-dir ./results

# Generate figures
python visualization/plot_results.py \
    --results ./results/evaluation_results.json \
    --output-dir ./figures
```

The evaluation includes:
- Leave-One-Day-Out cross-validation
- Link selection comparison (Top-k vs All-links vs Random)
- Wilcoxon signed-rank test for statistical significance
- Bootstrap confidence intervals
- Cohen's d effect size

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{anonymous2026lessismore,
  title={Less is More: The Dilution Effect in Multi-Link Wireless Sensing},
  author={Anonymous},
  booktitle={Under Review},
  year={2026},
  note={Paper under double-blind review}
}
```

**Note:** Citation will be updated with full author details upon paper acceptance.

## Acknowledgments

- ESP-IDF WiFi CSI API
- Seeed XIAO ESP32-C3 hardware platform
