#!/usr/bin/env python3
"""
Configuration constants for WiFi CSI TDMA System.

This is the open-source release of the ring-based TDMA scheduler
for WiFi Channel State Information (CSI) collection using ESP32 nodes.
"""

import os

# =============================================================================
# Network Configuration
# =============================================================================
UDP_PORT = 5000              # Server listens for CSI, HELLO, HEARTBEAT
TRIGGER_PORT = 5001          # ESP32s listen for trigger commands
BROADCAST_ADDR = "255.255.255.255"
BUFFER_SIZE = 4096           # UDP receive buffer

# =============================================================================
# TDMA Timing
# =============================================================================
# IMPORTANT: With N nodes, one TDMA cycle takes N * SLOT_DURATION_MS milliseconds.
# For 9 nodes at 20ms slots: 180ms cycle = 5.6 Hz sampling rate
# For motion detection (Nyquist for walking at 2 Hz): need >= 4 Hz
SLOT_DURATION_MS = 20        # Time per node slot
BURSTS_PER_TRIGGER = 10      # Number of packets per burst
CYCLE_DELAY_MS = 5           # Delay between TDMA cycles
MIN_NODES_TO_START = 2       # Minimum nodes to start scheduler

# =============================================================================
# Health Monitoring
# =============================================================================
HEARTBEAT_TIMEOUT_S = float(os.environ.get("HEARTBEAT_TIMEOUT_S", "30.0"))
HEARTBEAT_CHECK_INTERVAL_S = float(os.environ.get("HEARTBEAT_CHECK_S", "5.0"))
NODE_REMOVAL_TIMEOUT_S = float(os.environ.get("NODE_REMOVAL_S", "300.0"))

# =============================================================================
# Data Storage
# =============================================================================
DATA_DIR = os.environ.get("CSI_DATA_DIR", "./csi_data")
FILE_ROTATION_MINUTES = 60
WRITE_BUFFER_SIZE = 100

# =============================================================================
# Physical Map (optional)
# =============================================================================
PHYSICAL_MAP_FILE = os.environ.get("PHYSICAL_MAP", "./physical_map.txt")
