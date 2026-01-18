#!/usr/bin/env python3
"""
TDMA Scheduler - Orchestrates round-robin transmission triggers.

This is the open-source release of the ring-based TDMA scheduler
for WiFi Channel State Information (CSI) collection using ESP32 nodes.

Architecture:
    The scheduler implements a simple round-robin TDMA protocol:
    1. Each node gets a time slot to transmit
    2. All other nodes capture CSI from the transmitter
    3. After all nodes have transmitted, the cycle repeats

    With N nodes, one TDMA cycle takes N * SLOT_DURATION_MS milliseconds.
    For 9 nodes at 20ms slots: 180ms cycle = ~5.6 Hz effective sampling rate

Optimizations:
    - Cached node list (1s refresh) to avoid lock contention
    - Direct non-blocking socket sends (no thread pool)
    - Minimal registry updates for maximum throughput
"""

import asyncio
import socket
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List

from models import TriggerPacket
from config import (
    TRIGGER_PORT,
    SLOT_DURATION_MS,
    BURSTS_PER_TRIGGER,
    CYCLE_DELAY_MS,
    MIN_NODES_TO_START
)

logger = logging.getLogger(__name__)

# Node cache refresh interval (seconds)
NODE_CACHE_REFRESH_S = 1.0


class TDMAScheduler:
    """
    TDMA scheduler for WiFi CSI sensing mesh.

    Implements a round-robin protocol where each node transmits in turn
    while all other nodes capture the resulting CSI data.

    Example:
        scheduler = TDMAScheduler(registry)
        await scheduler.start()
        # ... scheduler runs automatically ...
        await scheduler.stop()
    """

    def __init__(self, registry, config: Optional[Dict] = None):
        """
        Initialize the TDMA scheduler.

        Args:
            registry: NodeRegistry instance for tracking nodes
            config: Optional configuration overrides
        """
        self.registry = registry
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._sock: Optional[socket.socket] = None

        # Current state
        self.current_tx_index = 0
        self.current_tx_mac: Optional[str] = None
        self.sequence_number = 0
        self.cycle_count = 0

        # Slot ID mapping for TX identification
        self._slot_id_map_maxlen = 64
        self.slot_id_map: Dict[int, str] = {}

        # Node cache (avoid lock contention)
        self._cached_nodes: List = []
        self._cache_time: float = 0.0

        # Pre-built broadcast address
        self._broadcast_addr = ("255.255.255.255", TRIGGER_PORT)

        # Statistics
        self.triggers_sent = 0
        self.trigger_errors = 0
        self.last_cycle_time: Optional[datetime] = None

    async def start(self):
        """Start the TDMA scheduler."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.setblocking(False)

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("TDMA Scheduler started")

    async def stop(self):
        """Stop the TDMA scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        if self._sock:
            self._sock.close()

        self.slot_id_map.clear()

        logger.info(f"TDMA Scheduler stopped. Sent {self.triggers_sent} triggers, "
                   f"{self.trigger_errors} errors, {self.cycle_count} cycles")

    async def _scheduler_loop(self):
        """
        Main TDMA scheduling loop.

        Continuously cycles through nodes, triggering each one to transmit
        while others capture CSI.
        """
        slot_time = SLOT_DURATION_MS / 1000.0
        cycle_delay = CYCLE_DELAY_MS / 1000.0

        while self._running:
            try:
                # Refresh node cache periodically
                now = time.time()
                if now - self._cache_time > NODE_CACHE_REFRESH_S:
                    self._cached_nodes = await self.registry.get_online_nodes()
                    self._cache_time = now

                nodes = self._cached_nodes

                if len(nodes) < MIN_NODES_TO_START:
                    logger.debug(f"Waiting for nodes: {len(nodes)}/{MIN_NODES_TO_START}")
                    await asyncio.sleep(1.0)
                    self._cache_time = 0  # Force refresh
                    continue

                # Reset index at start of new cycle
                if self.current_tx_index >= len(nodes):
                    self.current_tx_index = 0
                    self.cycle_count += 1
                    self.last_cycle_time = datetime.now()

                # Get current transmitter
                current_node = nodes[self.current_tx_index]
                self.current_tx_mac = current_node.mac

                # Send trigger (non-blocking)
                self._send_trigger(current_node)

                # Wait for transmission slot
                await asyncio.sleep(slot_time)

                # Move to next node
                self.current_tx_index += 1
                self.sequence_number += 1

                # Brief pause between slots
                if cycle_delay > 0:
                    await asyncio.sleep(cycle_delay)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                await asyncio.sleep(0.5)

    def _send_trigger(self, node):
        """
        Send trigger packet using non-blocking socket.

        Args:
            node: Node object with mac attribute
        """
        slot_id = self.current_tx_index
        self.slot_id_map[slot_id] = node.mac

        # Prevent unbounded growth
        if len(self.slot_id_map) > self._slot_id_map_maxlen:
            valid_slots = set(range(len(self._cached_nodes) if self._cached_nodes else 16))
            self.slot_id_map = {k: v for k, v in self.slot_id_map.items() if k in valid_slots}

        trigger = TriggerPacket(
            tx_node_mac=node.mac,
            seq=self.sequence_number,
            burst_count=BURSTS_PER_TRIGGER,
            slot_ms=SLOT_DURATION_MS,
            slot_id=slot_id
        )

        try:
            self._sock.sendto(trigger.to_bytes(), self._broadcast_addr)
            self.triggers_sent += 1
        except BlockingIOError:
            self.trigger_errors += 1
        except Exception as e:
            logger.error(f"Failed to send trigger: {e}")
            self.trigger_errors += 1

    def get_stats(self) -> dict:
        """Get scheduler statistics."""
        return {
            'running': self._running,
            'current_tx_mac': self.current_tx_mac,
            'sequence_number': self.sequence_number,
            'cycle_count': self.cycle_count,
            'triggers_sent': self.triggers_sent,
            'trigger_errors': self.trigger_errors,
            'last_cycle_time': self.last_cycle_time.isoformat() if self.last_cycle_time else None,
            'slot_duration_ms': SLOT_DURATION_MS,
            'bursts_per_trigger': BURSTS_PER_TRIGGER,
            'cached_nodes': len(self._cached_nodes)
        }

    def reset_stats(self):
        """Reset all statistics counters."""
        self.triggers_sent = 0
        self.trigger_errors = 0
        self.cycle_count = 0
        self.sequence_number = 0
        self.current_tx_index = 0
        self.last_cycle_time = None
        self.slot_id_map.clear()
        logger.info("Scheduler stats reset")

    def get_tx_mac_for_slot(self, slot_id: int) -> Optional[str]:
        """
        Resolve slot_id to TX MAC address.

        Args:
            slot_id: The slot identifier from a CSI batch packet

        Returns:
            TX MAC address if known, None otherwise
        """
        return self.slot_id_map.get(slot_id)
