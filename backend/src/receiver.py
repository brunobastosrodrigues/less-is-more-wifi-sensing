#!/usr/bin/env python3
"""
UDP Receiver - Async listener for incoming CSI and control packets.

Supports:
- Binary v1 protocol (128-byte CSI packets)
- Binary v2 protocol (148-byte CSI packets)
- Binary v5 protocol (batch packets)
- Binary HELLO/HEARTBEAT packets
- JSON fallback for legacy compatibility
"""

import asyncio
import json
import logging
import struct
from binascii import crc32
from datetime import datetime
from typing import Callable, Optional

from models import CSIPacket, PacketType
from config import UDP_PORT, BUFFER_SIZE

logger = logging.getLogger(__name__)

# Binary Protocol Constants
BINARY_PROTO_MAGIC = 0xC5

MSG_TYPE_HELLO = 0x01
MSG_TYPE_HEARTBEAT = 0x02
MSG_TYPE_CSI = 0x03
MSG_TYPE_CSI_V5 = 0x07
MSG_TYPE_BURST = 0x30
MSG_TYPE_TEST = 0xFF

BINARY_HELLO_SIZE = 40
BINARY_HEARTBEAT_SIZE = 24

CHIP_TYPES = {
    0x01: "ESP32",
    0x02: "ESP32-S2",
    0x03: "ESP32-C3",
    0x04: "ESP32-S3",
}


def validate_crc32(data: bytes) -> bool:
    """Validate CRC32 checksum."""
    if len(data) < 4:
        return False
    received_crc = struct.unpack('<I', data[-4:])[0]
    calculated_crc = crc32(data[:-4]) & 0xFFFFFFFF
    return received_crc == calculated_crc


def mac_bytes_to_str(mac_bytes: bytes) -> str:
    """Convert 6-byte MAC to colon-separated string."""
    return ':'.join(f'{b:02X}' for b in mac_bytes)


def ip_bytes_to_str(ip_bytes: bytes) -> str:
    """Convert 4-byte IP to dotted string."""
    return '.'.join(str(b) for b in ip_bytes)


class UDPProtocol(asyncio.DatagramProtocol):
    """asyncio UDP protocol handler."""

    def __init__(self, receiver: 'CSIReceiver'):
        self.receiver = receiver
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: tuple):
        asyncio.create_task(self.receiver._handle_packet(data, addr))

    def error_received(self, exc):
        logger.error(f"UDP error: {exc}")


class CSIReceiver:
    """
    Async UDP server that receives and routes packets from ESP32 nodes.
    """

    def __init__(self, registry, synchronizer=None):
        self.registry = registry
        self.synchronizer = synchronizer
        self.transport = None
        self.protocol = None

        self._on_csi: list[Callable] = []
        self._on_hello: list[Callable] = []
        self._on_heartbeat: list[Callable] = []

        # Statistics
        self.packets_received = 0
        self.packets_parsed = 0
        self.packets_failed = 0
        self.bytes_received = 0
        self.binary_packets = 0
        self.json_packets = 0

    async def start(self, host: str = '0.0.0.0', port: int = UDP_PORT):
        """Start the UDP server."""
        loop = asyncio.get_event_loop()
        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self),
            local_addr=(host, port)
        )
        logger.info(f"CSI Receiver listening on {host}:{port}")

    async def stop(self):
        """Stop the UDP server."""
        if self.transport:
            self.transport.close()
        logger.info("CSI Receiver stopped")

    def on_csi(self, callback: Callable):
        """Register callback for CSI packets."""
        self._on_csi.append(callback)

    def on_hello(self, callback: Callable):
        """Register callback for HELLO packets."""
        self._on_hello.append(callback)

    def on_heartbeat(self, callback: Callable):
        """Register callback for HEARTBEAT packets."""
        self._on_heartbeat.append(callback)

    async def _handle_packet(self, data: bytes, addr: tuple):
        """Parse and route incoming packet."""
        ip, port = addr
        self.packets_received += 1
        self.bytes_received += len(data)

        # Check for binary protocol
        if len(data) >= 2 and data[0] == BINARY_PROTO_MAGIC:
            msg_type = data[1]
            self.binary_packets += 1

            if msg_type == MSG_TYPE_HELLO and len(data) >= BINARY_HELLO_SIZE:
                if validate_crc32(data):
                    await self._handle_binary_hello(data, ip)
                    self.packets_parsed += 1
                return

            if msg_type == MSG_TYPE_HEARTBEAT and len(data) >= BINARY_HEARTBEAT_SIZE:
                if validate_crc32(data):
                    await self._handle_binary_heartbeat(data, ip)
                    self.packets_parsed += 1
                return

            if msg_type == MSG_TYPE_CSI_V5 or msg_type == MSG_TYPE_CSI:
                await self._handle_binary_csi(data, ip)
                self.packets_parsed += 1
                return

            if msg_type == MSG_TYPE_TEST:
                logger.info(f"Firmware boot detected from {ip}")
                self.packets_parsed += 1
                return

            if msg_type == MSG_TYPE_BURST:
                self.packets_parsed += 1
                return

            return

        # Fallback: JSON parsing
        try:
            payload = json.loads(data.decode('utf-8'))
            packet_type = payload.get('type', '').upper()
            self.json_packets += 1

            if packet_type == PacketType.HELLO.value:
                await self._handle_hello(payload, ip)
            elif packet_type == PacketType.HEARTBEAT.value:
                await self._handle_heartbeat(payload, ip)
            elif packet_type == PacketType.CSI.value:
                await self._handle_csi(payload, ip)

            self.packets_parsed += 1

        except json.JSONDecodeError:
            self.packets_failed += 1

    async def _handle_binary_csi(self, data: bytes, ip: str):
        """Handle binary CSI packet."""
        if not self.synchronizer:
            return

        result = self.synchronizer.process_packet(data, ip)
        if not result:
            return

        packets = result if isinstance(result, list) else [result]
        for packet in packets:
            await self.registry.update_link(packet.tx_mac, packet.rx_mac, packet.rssi)
            for callback in self._on_csi:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(packet)
                    else:
                        callback(packet)
                except Exception as e:
                    logger.error(f"CSI callback error: {e}")

    async def _handle_hello(self, payload: dict, ip: str):
        """Handle JSON HELLO packet."""
        mac = payload.get('mac', '').upper()
        chip = payload.get('chip', 'ESP32')
        version = payload.get('version', '1.0')

        if not mac:
            return

        node = await self.registry.register_node(mac, ip, chip, version)
        if self.synchronizer:
            self.synchronizer.register_node(mac, ip)

        logger.info(f"HELLO from {mac[-5:]} ({chip} v{version}) at {ip}")

        for callback in self._on_hello:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(node)
                else:
                    callback(node)
            except Exception as e:
                logger.error(f"HELLO callback error: {e}")

    async def _handle_heartbeat(self, payload: dict, ip: str):
        """Handle JSON HEARTBEAT packet."""
        mac = payload.get('mac', '').upper()
        uptime_s = payload.get('uptime_s', 0)
        free_heap = payload.get('free_heap', 0)

        if not mac:
            return

        await self.registry.update_heartbeat(mac, uptime_s, free_heap)

        for callback in self._on_heartbeat:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(mac, uptime_s, free_heap)
                else:
                    callback(mac, uptime_s, free_heap)
            except Exception as e:
                logger.error(f"HEARTBEAT callback error: {e}")

    async def _handle_binary_hello(self, data: bytes, ip: str):
        """Handle binary HELLO packet (40 bytes)."""
        try:
            mac = mac_bytes_to_str(data[2:8])
            chip_type = data[12]
            fw_major = data[13]
            fw_minor = data[14]
            fw_patch = data[15]
            fw_tag = data[16:24].rstrip(b'\x00').decode('ascii', errors='ignore')
            uptime_s, free_heap, min_heap = struct.unpack('<III', data[24:36])

            chip = CHIP_TYPES.get(chip_type, f"Unknown(0x{chip_type:02X})")
            version = f"{fw_major}.{fw_minor}.{fw_patch}-{fw_tag}"

            node = await self.registry.register_node(mac, ip, chip, version)
            if self.synchronizer:
                self.synchronizer.register_node(mac, ip)

            logger.info(f"HELLO from {mac[-5:]} ({chip} v{version}) at {ip} [binary]")

            for callback in self._on_hello:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(node)
                    else:
                        callback(node)
                except Exception as e:
                    logger.error(f"HELLO callback error: {e}")

        except Exception as e:
            logger.error(f"Binary HELLO parse error: {e}")

    async def _handle_binary_heartbeat(self, data: bytes, ip: str):
        """Handle binary HEARTBEAT packet (24 bytes)."""
        try:
            mac = mac_bytes_to_str(data[2:8])
            uptime_s, free_heap, min_heap = struct.unpack('<III', data[8:20])

            await self.registry.update_heartbeat(mac, uptime_s, free_heap)

            for callback in self._on_heartbeat:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(mac, uptime_s, free_heap)
                    else:
                        callback(mac, uptime_s, free_heap)
                except Exception as e:
                    logger.error(f"HEARTBEAT callback error: {e}")

        except Exception as e:
            logger.error(f"Binary HEARTBEAT parse error: {e}")

    async def _handle_csi(self, payload: dict, ip: str):
        """Handle JSON CSI packet (legacy)."""
        try:
            packet = CSIPacket(
                tx_mac=payload.get('tx_mac', '').upper(),
                rx_mac=payload.get('rx_mac', '').upper(),
                rx_ip=ip,
                seq=payload.get('seq', 0),
                rssi=payload.get('rssi', 0),
                noise_floor=payload.get('noise_floor', -90),
                channel=payload.get('channel', 0),
                bandwidth=payload.get('bw', 20),
                timestamp_us=payload.get('timestamp_us', 0),
                csi_len=payload.get('csi_len', 0),
                csi_raw=payload.get('csi_data', [])
            )

            await self.registry.update_link(packet.tx_mac, packet.rx_mac, packet.rssi)

            for callback in self._on_csi:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(packet)
                    else:
                        callback(packet)
                except Exception as e:
                    logger.error(f"CSI callback error: {e}")

        except Exception as e:
            logger.error(f"CSI packet parse error: {e}")

    def get_stats(self) -> dict:
        """Get receiver statistics."""
        return {
            'packets_received': self.packets_received,
            'packets_parsed': self.packets_parsed,
            'packets_failed': self.packets_failed,
            'bytes_received': self.bytes_received,
            'binary_packets': self.binary_packets,
            'json_packets': self.json_packets,
        }
