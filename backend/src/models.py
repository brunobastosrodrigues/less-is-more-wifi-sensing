#!/usr/bin/env python3
"""
Data models for WiFi CSI TDMA System.

Dataclasses for nodes, packets, and system state.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional
import numpy as np


class NodeStatus(Enum):
    ONLINE = "online"
    OFFLINE = "offline"
    TRANSMITTING = "transmitting"


class PacketType(Enum):
    HELLO = "HELLO"
    HEARTBEAT = "HEARTBEAT"
    CSI = "CSI"
    TRIGGER = "TRIGGER"


@dataclass
class Node:
    """Represents an ESP32 node in the mesh."""
    mac: str
    ip: str
    chip: str = "ESP32"
    version: str = "1.0"
    status: NodeStatus = NodeStatus.OFFLINE
    last_seen: datetime = field(default_factory=datetime.now)
    uptime_s: int = 0
    free_heap: int = 0
    position: Optional[tuple] = None  # (x, y, z) from physical map
    packets_received: int = 0
    packets_sent: int = 0

    def update_heartbeat(self, uptime_s: int, free_heap: int):
        """Update node with heartbeat data."""
        self.last_seen = datetime.now()
        self.uptime_s = uptime_s
        self.free_heap = free_heap
        if self.status == NodeStatus.OFFLINE:
            self.status = NodeStatus.ONLINE

    def is_stale(self, timeout_s: float) -> bool:
        """Check if node hasn't been seen within timeout."""
        elapsed = (datetime.now() - self.last_seen).total_seconds()
        return elapsed > timeout_s


@dataclass
class CSIPacket:
    """Channel State Information packet from a receiver node."""
    tx_mac: str
    rx_mac: str
    rx_ip: str
    seq: int
    rssi: int
    noise_floor: int
    channel: int
    bandwidth: int
    timestamp_us: int
    csi_len: int
    csi_raw: List[int]  # Raw I/Q interleaved
    received_at: datetime = field(default_factory=datetime.now)

    def to_complex(self) -> np.ndarray:
        """Convert raw I/Q data to complex numpy array."""
        iq = np.array(self.csi_raw, dtype=np.float32)
        num_subcarriers = len(iq) // 2
        imag = iq[0::2]  # Even indices: imaginary
        real = iq[1::2]  # Odd indices: real
        return real + 1j * imag

    def amplitude(self) -> np.ndarray:
        """Get amplitude of each subcarrier."""
        return np.abs(self.to_complex())

    def phase(self) -> np.ndarray:
        """Get phase of each subcarrier."""
        return np.angle(self.to_complex())

    def to_dict(self) -> dict:
        """Serialize for JSON storage."""
        return {
            "tx_mac": self.tx_mac,
            "rx_mac": self.rx_mac,
            "seq": self.seq,
            "rssi": self.rssi,
            "noise_floor": self.noise_floor,
            "channel": self.channel,
            "bw": self.bandwidth,
            "timestamp_us": self.timestamp_us,
            "csi_len": self.csi_len,
            "csi_data": self.csi_raw,
            "received_at": self.received_at.isoformat()
        }


@dataclass
class TriggerPacket:
    """
    Trigger command sent to ESP32 to initiate burst.

    Binary format (32 bytes):
    - magic (1 byte): 0xC5
    - type (1 byte): 0x10 = TRIGGER
    - tx_mac (6 bytes): MAC of transmitting node
    - seq (4 bytes): uint32 sequence number
    - burst_count (2 bytes): uint16
    - slot_ms (2 bytes): uint16 slot duration
    - samples_per_meas (1 byte): samples to collect
    - send_all_samples (1 byte): 0=best only, 1=send all
    - slot_id (1 byte): slot identifier (0-255)
    - reserved (9 bytes)
    - crc32 (4 bytes)
    """
    tx_node_mac: str
    seq: int
    burst_count: int
    slot_ms: int = 80
    samples_per_meas: int = 5
    send_all_samples: bool = False
    slot_id: int = 0

    def _mac_to_bytes(self, mac: str) -> bytes:
        """Convert MAC string to 6 bytes."""
        return bytes(int(b, 16) for b in mac.split(':'))

    def to_bytes(self) -> bytes:
        """Serialize to binary format (32 bytes)."""
        import struct
        from binascii import crc32

        packet = bytearray(32)
        packet[0] = 0xC5  # Magic
        packet[1] = 0x10  # MSG_TYPE_TRIGGER
        packet[2:8] = self._mac_to_bytes(self.tx_node_mac)
        struct.pack_into('<I', packet, 8, self.seq)
        struct.pack_into('<H', packet, 12, self.burst_count)
        struct.pack_into('<H', packet, 14, self.slot_ms)
        packet[16] = self.samples_per_meas
        packet[17] = 1 if self.send_all_samples else 0
        packet[18] = self.slot_id & 0xFF

        # Calculate and append CRC32
        crc = crc32(bytes(packet[:-4])) & 0xFFFFFFFF
        struct.pack_into('<I', packet, 28, crc)

        return bytes(packet)


@dataclass
class LinkStats:
    """Statistics for a TX->RX link pair."""
    tx_mac: str
    rx_mac: str
    packet_count: int = 0
    last_rssi: int = 0
    avg_rssi: float = 0.0
    last_seen: datetime = field(default_factory=datetime.now)
    rssi_history: list = field(default_factory=list)
    rssi_variance: float = 0.0

    def update(self, rssi: int):
        """Update stats with new packet."""
        self.packet_count += 1
        self.last_rssi = rssi
        alpha = 0.1
        self.avg_rssi = alpha * rssi + (1 - alpha) * self.avg_rssi
        self.last_seen = datetime.now()

        self.rssi_history.append(rssi)
        if len(self.rssi_history) > 30:
            self.rssi_history.pop(0)

        if len(self.rssi_history) >= 5:
            mean = sum(self.rssi_history) / len(self.rssi_history)
            self.rssi_variance = sum((x - mean) ** 2 for x in self.rssi_history) / len(self.rssi_history)
