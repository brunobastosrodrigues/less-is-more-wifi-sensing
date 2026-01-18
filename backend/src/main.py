#!/usr/bin/env python3
"""
WiFi CSI TDMA System - Main Entry Point

This is the open-source release of the ring-based TDMA scheduler
for WiFi Channel State Information (CSI) collection using ESP32 nodes.

Usage:
    python main.py

The system will:
1. Start listening for ESP32 node registrations on UDP port 5000
2. Once MIN_NODES_TO_START nodes are online, begin TDMA scheduling
3. Broadcast trigger commands on UDP port 5001
4. Collect and synchronize CSI data from all nodes

Press Ctrl+C to stop.
"""

import asyncio
import logging
import signal
import sys
from datetime import datetime

from registry import NodeRegistry
from synchronizer import FrameSynchronizer
from receiver import CSIReceiver
from scheduler import TDMAScheduler
from config import UDP_PORT, MIN_NODES_TO_START

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class CSIServer:
    """
    Main server coordinating TDMA scheduling and CSI collection.
    """

    def __init__(self):
        # Core components
        self.registry = NodeRegistry()
        self.synchronizer = FrameSynchronizer(frame_rate=20)
        self.receiver = CSIReceiver(self.registry, self.synchronizer)
        self.scheduler = TDMAScheduler(self.registry)

        # Link synchronizer to scheduler for slot_id resolution
        self.synchronizer._scheduler = self.scheduler

        # Statistics
        self._running = False
        self._status_task = None

    async def start(self):
        """Start all server components."""
        logger.info("=" * 60)
        logger.info("  WiFi CSI TDMA System")
        logger.info("  Ring-Based TDMA Scheduler for ESP32 Nodes")
        logger.info("=" * 60)

        # Start components
        await self.registry.start()
        await self.receiver.start(port=UDP_PORT)
        await self.synchronizer.start()
        await self.scheduler.start()

        self._running = True
        self._status_task = asyncio.create_task(self._status_loop())

        logger.info("")
        logger.info(f"Server ready on UDP port {UDP_PORT}")
        logger.info(f"Waiting for {MIN_NODES_TO_START}+ ESP32 nodes to come online...")
        logger.info("Press Ctrl+C to stop")
        logger.info("")

    async def stop(self):
        """Stop all server components."""
        self._running = False

        if self._status_task:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass

        await self.scheduler.stop()
        await self.synchronizer.stop()
        await self.receiver.stop()
        await self.registry.stop()

        logger.info("Server stopped")

    async def _status_loop(self):
        """Print periodic status updates."""
        while self._running:
            try:
                await asyncio.sleep(10)

                if not self._running:
                    break

                # Get stats
                online, total = await self.registry.get_node_count()
                sched_stats = self.scheduler.get_stats()
                recv_stats = self.receiver.get_stats()
                sync_stats = self.synchronizer.get_stats()

                logger.info(
                    f"Status: nodes={online}/{total}, "
                    f"cycles={sched_stats['cycle_count']}, "
                    f"triggers={sched_stats['triggers_sent']}, "
                    f"packets={recv_stats['packets_received']}, "
                    f"frames={sync_stats['frames_built']}"
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Status loop error: {e}")


async def main():
    """Main entry point."""
    server = CSIServer()

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("\nShutdown requested...")
        asyncio.create_task(shutdown())

    async def shutdown():
        await server.stop()
        loop.stop()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await server.start()
        # Run forever
        while server._running:
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        await server.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
