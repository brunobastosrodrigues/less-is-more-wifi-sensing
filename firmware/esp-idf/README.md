# ESP32-C3 WiFi CSI Firmware

ESP-IDF firmware for WiFi Channel State Information (CSI) capture on XIAO ESP32-C3 boards.

## Current Version

**v5.1.0-PURE** (January 2026)

## IMPORTANT: Configuration Required

Before building, you **MUST** configure your WiFi credentials and server IP:

### Option 1: Edit `sdkconfig.defaults`

```
CONFIG_WIFI_SSID="your_wifi_ssid"
CONFIG_WIFI_PASSWORD="your_wifi_password"
CONFIG_SERVER_IP="192.168.1.100"  # IP of your backend server
```

### Option 2: Use menuconfig

```bash
idf.py menuconfig
# Navigate to: WiFi Tomography Node Configuration
# Set: WiFi SSID, WiFi Password, Server IP Address
```

**Note:** All ESP32 nodes and the backend server must be on the same WiFi network.

## Directory Structure

```
firmware/esp-idf/
├── main/
│   ├── main.c           # Main firmware source (see header for version history)
│   └── CMakeLists.txt   # Component configuration
├── CMakeLists.txt       # Project configuration
├── sdkconfig.defaults   # Default build configuration
├── dependencies.lock    # ESP-IDF dependencies
└── README.md            # This file
```

## Documentation

All firmware documentation is in `docs/firmware/`:

- `BUILD_NOTES.md` - General build notes and configuration
- `BUILD_WINDOWS.md` - Windows-specific build procedure
- `DEBUG_TRIGGER_CRASH.md` - Debug notes for v4.1.2 stack overflow fix
- `FIRMWARE_v4.1.3_OPTIMIZATIONS.md` - Optimization guide for v4.1.3

## Quick Build

### Prerequisites
- ESP-IDF v5.5.2
- Python 3.11
- CMake, Ninja

### Build Commands

```bash
# Clean build
rm -rf build sdkconfig

# Set target and build
idf.py set-target esp32c3
idf.py build

# Flash
idf.py -p COMx flash monitor
```

For Windows-specific instructions, see `docs/firmware/BUILD_WINDOWS.md`.

## Version History

See `main/main.c` header for complete version history.

| Version | Tag | Key Changes |
|---------|-----|-------------|
| 4.1.3 | OPTIMIZED | Trigger rate limiting, reduced burst count |
| 4.1.2 | STABLE | Fixed stack overflow in trigger processing |
| 4.1.1 | PUREBIN | Faster burst timing (200µs delay) |
| 4.1.0 | PUREBIN | Removed cJSON, pure binary protocol |
| 4.0 | BINARY | Zero-malloc binary protocol |
| 3.6 | RELIABLE | Task watchdog, heap monitoring |
