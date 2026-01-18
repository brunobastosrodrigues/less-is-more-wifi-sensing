# TDMA Timing Design for ESP32-C3 WiFi Sensing

## Executive Summary

This document analyzes the practical constraints of TDMA timing for distributed WiFi sensing on resource-constrained ESP32-C3 hardware.

**Key Finding**: The backend controls all timing through trigger packets. No firmware changes are required to experiment with different TDMA parameters.

---

## 1. System Architecture

### 1.1 Timing Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         TDMA Cycle for N Nodes                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Backend          Firmware (per node)                                   │
│  ────────         ──────────────────                                    │
│                                                                         │
│  t=0ms   ──TRIGGER(TX=Node1)──▶  Node1: transmit_burst(N packets)      │
│                                  Others: capture CSI in ring buffer     │
│                                                                         │
│  t=Slot  ──TRIGGER(TX=Node2)──▶  Node2: transmit_burst(N packets)      │
│                                  Others: send CSI, then capture new     │
│                                                                         │
│  ...repeats for all nodes...                                            │
│                                                                         │
│  t=N*Slot ─────────────────────▶  One TDMA cycle complete              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Default Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `SLOT_DURATION_MS` | 20 ms | Per-node transmission window |
| `BURSTS_PER_TRIGGER` | 10 | Packets per transmission |
| `CYCLE_DELAY_MS` | 5 ms | Pause between slots |
| `MIN_NODES_TO_START` | 2 | Minimum for TDMA operation |

**Effective sampling rate with 9 nodes:**
- Cycle time = 9 × (20 + 5) = 225 ms
- Sampling rate = 1000/225 = **~4.4 Hz**

---

## 2. ESP32-C3 Hardware Constraints

### 2.1 Known Limitations

| Resource | Capacity | Usage Notes |
|----------|----------|-------------|
| SRAM | 400 KB (320 KB usable) | Ring buffer, stack, WiFi buffers |
| CPU | Single core, 160 MHz | No true parallelism |
| WiFi | Half-duplex | Cannot TX and RX simultaneously |
| Stack | 4 KB per task | Safe minimum for stability |

### 2.2 Minimum Safe Margins

| Resource | Safe Margin | Rationale |
|----------|-------------|-----------|
| Heap | > 8 KB free | Below this triggers warning |
| Heap (critical) | > 4 KB free | Below this triggers reboot |
| Stack per task | 4 KB | Proved stable in production |
| Watchdog | 30 seconds | Auto-reboot on hang |

---

## 3. Sampling Rate vs. Feature Feasibility

| Feature | Required Rate | 9-Node @ 20ms | 4-Node @ 20ms |
|---------|---------------|---------------|---------------|
| Presence | 1-2 Hz | ✅ 4 Hz | ✅ 10 Hz |
| Zone tracking | 2-4 Hz | ✅ 4 Hz | ✅ 10 Hz |
| Breathing | 2-3 Hz | ✅ 4 Hz | ✅ 10 Hz |
| Walking direction | 4-6 Hz | ⚠️ 4 Hz | ✅ 10 Hz |
| Gesture | 10-20 Hz | ❌ | ❌ |

---

## 4. Recommended Configuration

### 4.1 Conservative (Default)

```python
SLOT_DURATION_MS = 20
BURSTS_PER_TRIGGER = 10
CYCLE_DELAY_MS = 5
MIN_NODES_TO_START = 2
```

### 4.2 Aggressive (Test First!)

```python
SLOT_DURATION_MS = 10
BURSTS_PER_TRIGGER = 10
CYCLE_DELAY_MS = 1
MIN_NODES_TO_START = 2
```

---

## 5. Sampling Rate Formula

```
Rate (Hz) = 1000 / (N × (slot_ms + cycle_delay_ms))

Where:
  N = number of active nodes
  slot_ms = SLOT_DURATION_MS
  cycle_delay_ms = CYCLE_DELAY_MS
```

### Quick Reference Table

| Nodes | Slot (ms) | Delay (ms) | Rate (Hz) |
|-------|-----------|------------|-----------|
| 9 | 20 | 5 | 4.4 |
| 7 | 20 | 5 | 5.7 |
| 4 | 20 | 5 | 10.0 |
| 3 | 20 | 5 | 13.3 |
| 9 | 10 | 1 | 9.1 |
| 4 | 10 | 1 | 22.7 |
