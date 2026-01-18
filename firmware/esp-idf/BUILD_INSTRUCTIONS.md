# Firmware Build Instructions for v5.1.0-PURE

## What Has Been Done

### Backend Optimizations (Already Deployed)
The backend has been optimized for high-frequency TDMA operation:
- **P1-P4**: Scheduler optimizations (node caching, direct socket send, skip status updates)
- **P5**: Frame rate increased from 20Hz to 50Hz
- **P6**: Non-blocking pipeline processing with frame dropping
- **Config**: SLOT_DURATION_MS=20ms, CYCLE_DELAY_MS=5ms

Result: 5x improvement in cycle rate (0.54 Hz → 2.7 Hz Nyquist)

### Firmware Optimizations (Ready to Flash)
The firmware in `main/main.c` includes these optimizations:

| Optimization | Change | Impact |
|--------------|--------|--------|
| OPT-1 | Binary burst packet protocol (0x30) | Reduced parsing overhead |
| OPT-2 | Pre-computed MAC bytes | Eliminates per-packet MAC parsing |
| OPT-3 | Inline pilot phase extraction | Reduces function call overhead |
| OPT-4 | Socket timeout 500→200ms | Faster trigger response |
| OPT-5 | Mutex timeout 100→10ms | Faster socket acquisition |
| OPT-6 | Heartbeat stack 4096→2048 bytes | More heap headroom |
| OPT-7 | DEBUG level for HELLO/burst logs | Reduced logging overhead |
| P0 | MIN_TRIGGER_INTERVAL_MS 300→50ms | Allows higher sampling rates |

### Protocol v5.0 Support (Optional)
The firmware also includes Protocol v5.0 "Adaptive Streaming" support:
- 8-byte batch header instead of 148-byte per-packet
- Adaptive batching with 3ms silence timeout
- Controlled via Kconfig: `CONFIG_PROTOCOL_V5_ENABLED`
- **Default: DISABLED** (use v4.x protocol for now)

---

## Build Instructions

### Prerequisites
- ESP-IDF v5.5.2 environment configured
- ESP32-C3 boards connected via USB

### Step 1: Clean Build Environment
```bash
cd firmware/esp-idf
rm -f sdkconfig
```

**CRITICAL**: You MUST delete `sdkconfig` before building. The `sdkconfig.defaults` contains the correct settings, but an existing `sdkconfig` will override them.

### Step 2: Build Firmware
```bash
idf.py build
```

Expected output:
- Firmware version: 4.1.3 (or 5.0.0 if v5.0 enabled)
- Firmware tag: OPTIMIZED (or STREAM if v5.0 enabled)

### Step 3: Flash First Test Board
```bash
idf.py -p /dev/ttyACM0 flash monitor
```

### Step 4: Verify First Board
Watch the serial monitor for:
1. `[BOOT] Firmware: 4.1.3-OPTIMIZED` (or 5.0.0-STREAM)
2. `[WIFI] Connected to AP`
3. `[HELLO] Sent binary HELLO`
4. `[TRIGGER] Received trigger` messages

Check backend logs:
```bash
docker compose logs -f server 2>&1 | grep -E "(HELLO|CSI|cycle)"
```

Expected: Board should appear online within 10 seconds

### Step 5: Flash Remaining Boards
Once the first board is verified, flash the remaining 8 boards one by one:

```bash
# For each board:
idf.py -p /dev/ttyACM0 flash  # or whichever port
```

---

## Verification Checklist

After all boards are flashed:

1. **All nodes online**: Check backend API
   ```bash
   curl -s http://localhost:8000/api/stats | jq '.nodes_online'
   # Expected: 9
   ```

2. **Cycle rate improved**: Check cycles per second
   ```bash
   curl -s http://localhost:8000/api/stats | jq '.cycle_count, .uptime_s'
   # Calculate: cycle_count / uptime_s should be > 2.5 Hz
   ```

3. **No packet errors**: Check parse success
   ```bash
   curl -s http://localhost:8000/api/stats | jq '.packets_received, .packets_parsed'
   # Both numbers should be equal (100% parse success)
   ```

---

## Rollback Procedure

If issues occur, revert to previous firmware:

```bash
git checkout HEAD~1 -- firmware/esp-idf/main/main.c
git checkout HEAD~1 -- firmware/esp-idf/sdkconfig.defaults
rm -f sdkconfig
idf.py build flash
```

---

## Network Configuration

The firmware is pre-configured for the current network:
- SSID: Set in Kconfig
- Server IP: Set in Kconfig
- UDP Port: 5000 (CSI/HELLO/HEARTBEAT)
- Trigger Port: 5001

If network settings need changes, run `idf.py menuconfig` before building.
