#!/usr/bin/env python3
"""
Node Registry - Tracks ESP32 nodes in the mesh.

Handles discovery, heartbeats, and health monitoring.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Callable
from pathlib import Path

from models import Node, NodeStatus, LinkStats
from config import (
    HEARTBEAT_TIMEOUT_S,
    HEARTBEAT_CHECK_INTERVAL_S,
    NODE_REMOVAL_TIMEOUT_S,
    PHYSICAL_MAP_FILE
)

logger = logging.getLogger(__name__)


class NodeRegistry:
    """
    Central registry for all ESP32 nodes in the mesh.
    Thread-safe via asyncio locks.
    """

    def __init__(self):
        self._nodes: Dict[str, Node] = {}
        self._links: Dict[tuple, LinkStats] = {}
        self._lock = asyncio.Lock()
        self._positions: Dict[str, tuple] = {}
        self._on_node_change: List[Callable] = []
        self._health_task: Optional[asyncio.Task] = None

    async def start(self):
        """Start the registry and load physical map."""
        self._load_physical_map()
        self._health_task = asyncio.create_task(self._health_check_loop())
        logger.info("Node registry started")

    async def stop(self):
        """Stop health monitoring."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
        logger.info("Node registry stopped")

    def _load_physical_map(self):
        """Load node positions from physical_map.txt."""
        path = Path(PHYSICAL_MAP_FILE)
        if not path.exists():
            logger.warning(f"Physical map not found: {path}")
            return

        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 4:
                        mac = parts[0].strip().upper()
                        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                        self._positions[mac] = (x, y, z)
            logger.info(f"Loaded {len(self._positions)} node positions")
        except Exception as e:
            logger.error(f"Failed to load physical map: {e}")

    async def register_node(self, mac: str, ip: str, chip: str = "ESP32",
                           version: str = "1.0") -> Node:
        """Register a new node or update existing one."""
        mac = mac.upper()
        async with self._lock:
            if mac in self._nodes:
                node = self._nodes[mac]
                node.ip = ip
                node.chip = chip
                node.version = version
                node.last_seen = datetime.now()
                if node.status == NodeStatus.OFFLINE:
                    node.status = NodeStatus.ONLINE
                    logger.info(f"Node {mac} came back online at {ip}")
            else:
                node = Node(
                    mac=mac,
                    ip=ip,
                    chip=chip,
                    version=version,
                    status=NodeStatus.ONLINE,
                    position=self._positions.get(mac)
                )
                self._nodes[mac] = node
                logger.info(f"New node registered: {mac} ({chip}) at {ip}")

        await self._notify_change()
        return node

    async def update_heartbeat(self, mac: str, uptime_s: int, free_heap: int):
        """Update node with heartbeat data."""
        mac = mac.upper()
        async with self._lock:
            if mac in self._nodes:
                self._nodes[mac].update_heartbeat(uptime_s, free_heap)

    async def update_link(self, tx_mac: str, rx_mac: str, rssi: int):
        """Update link statistics between two nodes."""
        tx_mac = tx_mac.upper()
        rx_mac = rx_mac.upper()
        key = (tx_mac, rx_mac)

        async with self._lock:
            if key not in self._links:
                self._links[key] = LinkStats(tx_mac=tx_mac, rx_mac=rx_mac)
            self._links[key].update(rssi)

            if rx_mac in self._nodes:
                self._nodes[rx_mac].packets_received += 1
                self._nodes[rx_mac].last_seen = datetime.now()
                if self._nodes[rx_mac].status == NodeStatus.OFFLINE:
                    self._nodes[rx_mac].status = NodeStatus.ONLINE

            if tx_mac in self._nodes:
                self._nodes[tx_mac].last_seen = datetime.now()
                if self._nodes[tx_mac].status == NodeStatus.OFFLINE:
                    self._nodes[tx_mac].status = NodeStatus.ONLINE

    async def set_transmitting(self, mac: str):
        """Mark a node as currently transmitting."""
        mac = mac.upper()
        async with self._lock:
            if mac in self._nodes:
                self._nodes[mac].status = NodeStatus.TRANSMITTING
                self._nodes[mac].packets_sent += 1

    async def clear_transmitting(self, mac: str):
        """Clear transmitting status."""
        mac = mac.upper()
        async with self._lock:
            if mac in self._nodes:
                if self._nodes[mac].status == NodeStatus.TRANSMITTING:
                    self._nodes[mac].status = NodeStatus.ONLINE

    async def get_online_nodes(self) -> List[Node]:
        """Get list of all online nodes, sorted by MAC for consistent ordering."""
        async with self._lock:
            nodes = [n for n in self._nodes.values()
                    if n.status in (NodeStatus.ONLINE, NodeStatus.TRANSMITTING)]
            return sorted(nodes, key=lambda n: n.mac)

    async def get_all_nodes(self) -> List[Node]:
        """Get all registered nodes."""
        async with self._lock:
            return list(self._nodes.values())

    async def get_node(self, mac: str) -> Optional[Node]:
        """Get a specific node by MAC."""
        mac = mac.upper()
        async with self._lock:
            return self._nodes.get(mac)

    async def get_link_stats(self) -> Dict[tuple, LinkStats]:
        """Get all link statistics."""
        async with self._lock:
            return dict(self._links)

    async def get_node_count(self) -> tuple:
        """Returns (online_count, total_count)."""
        async with self._lock:
            total = len(self._nodes)
            online = sum(1 for n in self._nodes.values()
                        if n.status != NodeStatus.OFFLINE)
            return online, total

    def on_change(self, callback: Callable):
        """Register callback for node state changes."""
        self._on_node_change.append(callback)

    async def _notify_change(self):
        """Notify all registered callbacks."""
        for callback in self._on_node_change:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def _health_check_loop(self):
        """Periodically check for stale nodes."""
        while True:
            try:
                await asyncio.sleep(HEARTBEAT_CHECK_INTERVAL_S)
                await self._check_stale_nodes()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_stale_nodes(self):
        """Mark nodes that haven't sent heartbeats as offline."""
        changed = False
        async with self._lock:
            to_remove = []

            for mac, node in self._nodes.items():
                if node.status != NodeStatus.OFFLINE:
                    if node.is_stale(HEARTBEAT_TIMEOUT_S):
                        logger.warning(f"Node {node.mac} went offline (timeout)")
                        node.status = NodeStatus.OFFLINE
                        changed = True
                else:
                    if node.is_stale(NODE_REMOVAL_TIMEOUT_S):
                        to_remove.append(mac)

            for mac in to_remove:
                del self._nodes[mac]
                logger.info(f"Removed stale node: {mac}")
                changed = True

        if changed:
            await self._notify_change()
