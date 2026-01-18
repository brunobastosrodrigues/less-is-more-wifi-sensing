/**
 * WiFi Tomography Node - ESP-IDF with True CSI
 * For: ESP32-C3 (Seeed XIAO)
 *
 * Optimized for ESP-IDF v5.x with:
 * - IRAM placement for time-critical code
 * - Ring buffer for CSI data (no missed packets)
 * - Persistent sockets (avoid create/close overhead)
 * - Disabled power save for lowest latency
 * - Full binary protocol (zero malloc overhead)
 *
 * ============================================================================
 * VERSION: 5.1.0-PURE
 * ============================================================================
 *
 * Features:
 * - Zero-malloc binary protocol (no JSON)
 * - IRAM-placed CSI callback for lowest latency
 * - Spinlock-protected trigger state (ISR-safe)
 * - Task watchdog with 30s timeout
 * - WiFi auto-reconnect with exponential backoff
 * - Batch CSI streaming (up to 12 samples/packet)
 * - CRC32 validated packets
 * - 50 Hz sustained frame rate (Nyquist validated)
 *
 * ============================================================================
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <stdatomic.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "freertos/semphr.h"
#include "freertos/ringbuf.h"
#include "esp_system.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "lwip/err.h"
#include "lwip/sys.h"
#include "lwip/sockets.h"
// Pure binary protocol - zero JSON/malloc overhead
#include "esp_crc.h"
#include "rom/ets_sys.h"  // For esp_rom_delay_us
#include "esp_task_wdt.h"  // Task watchdog

// =============================================================================
// RELIABILITY CONFIGURATION
// =============================================================================
#define WATCHDOG_TIMEOUT_S      30      // Watchdog timeout in seconds
#define HEAP_WARNING_THRESHOLD  8192    // Warn if heap drops below this
#define HEAP_CRITICAL_THRESHOLD 4096    // Reboot if heap drops below this
#define WIFI_RECONNECT_DELAY_MS 5000    // Delay between WiFi reconnection attempts
#define SOCKET_TIMEOUT_MS       100     // Socket receive timeout (reduced from 5000ms for faster response)

// =============================================================================
// CONFIGURATION - Use menuconfig or override here
// =============================================================================
#ifndef CONFIG_WIFI_SSID
#define CONFIG_WIFI_SSID "your_wifi_ssid"
#endif

#ifndef CONFIG_WIFI_PASSWORD
#define CONFIG_WIFI_PASSWORD "your_wifi_password"
#endif

#ifndef CONFIG_SERVER_IP
#define CONFIG_SERVER_IP "192.168.1.100"  // IP of the machine running the backend
#endif

#ifndef CONFIG_SERVER_PORT
#define CONFIG_SERVER_PORT 5000
#endif

#ifndef CONFIG_TRIGGER_PORT
#define CONFIG_TRIGGER_PORT 5001
#endif

#ifndef CONFIG_BURST_PORT
#define CONFIG_BURST_PORT 5555
#endif

// WiFi channel to force (0 = auto, 1-13 = specific channel)
// Set this to ensure all nodes connect to the same AP on the same channel
#ifndef CONFIG_WIFI_CHANNEL
#define CONFIG_WIFI_CHANNEL 11
#endif

// Number of burst packets to transmit (backend typically sends 10)
#ifndef CONFIG_BURST_COUNT
#define CONFIG_BURST_COUNT 100
#endif

// Delay between burst packets in microseconds
// - PHY minimum: ~100 µs (preamble + DIFS)
// - Safe range: 150-500 µs
// - Default: 200 µs (2x PHY minimum for reliability margin)
// - With 10 packets: 10 × 200µs = 2ms burst time
#ifndef CONFIG_BURST_DELAY_US
#define CONFIG_BURST_DELAY_US 200
#endif

// =============================================================================
// CONSTANTS
// =============================================================================
static const char *TAG = "CSI_NODE";
#define MAX_CSI_DATA_LEN     384
#define CSI_RINGBUF_SIZE     (MAX_CSI_DATA_LEN * 16)  // Larger buffer for multi-sample
#define HEARTBEAT_INTERVAL_MS 2000
#define ANNOUNCE_INTERVAL_MS  5000
#define WIFI_RETRY_MAX        10

// Trigger rate limiting - ignore triggers that arrive too fast
// Reduced from 300ms to allow higher sampling rates with optimized backend
#define MIN_TRIGGER_INTERVAL_MS 50   // Minimum time between processed triggers

// Pilot subcarrier indices for HT20 (in 52-subcarrier array, 0-indexed)
// These correspond to subcarriers -21, -7, +7, +21 relative to center
#define PILOT_SC_1  6   // subcarrier -21
#define PILOT_SC_2  20  // subcarrier -7
#define PILOT_SC_3  32  // subcarrier +7
#define PILOT_SC_4  46  // subcarrier +21

// WiFi event group bits
#define WIFI_CONNECTED_BIT    BIT0
#define WIFI_FAIL_BIT         BIT1

// =============================================================================
// CSI DATA STRUCTURE (packed for ring buffer) - Enhanced with signal info
// =============================================================================
typedef struct __attribute__((packed)) {
    int64_t timestamp_us;
    int8_t  rssi;
    int8_t  noise_floor;
    uint8_t channel;
    uint8_t secondary_channel;
    uint8_t rate;
    uint8_t sig_mode;           // 0=non-HT, 1=HT, 3=VHT
    uint8_t cwb;                // Channel bandwidth (0=20MHz, 1=40MHz)
    uint8_t aggregation;
    uint8_t stbc;
    uint8_t fec_coding;
    uint8_t sgi;                // Short Guard Interval
    uint8_t rx_state;           // RX state info
    uint16_t len;
    uint8_t src_mac[6];
    int8_t  data[MAX_CSI_DATA_LEN];
} csi_packet_t;

// =============================================================================
// BINARY PROTOCOL - Zero-malloc, fixed-size packets for all messages
// =============================================================================
// Protocol magic byte (first byte of all packets)
#define BINARY_PROTO_MAGIC      0xC5

// Message types (second byte)
#define MSG_TYPE_HELLO          0x01    // Firmware → Server: Node registration
#define MSG_TYPE_HEARTBEAT      0x02    // Firmware → Server: Keepalive
#define MSG_TYPE_CSI            0x03    // Firmware → Server: CSI data (v2)
#define MSG_TYPE_TRIGGER        0x10    // Server → Firmware: Start measurement
#define MSG_TYPE_DISCOVER       0x11    // Server → Firmware: Request HELLO
#define MSG_TYPE_TEST           0xFF    // Connectivity test
#define MSG_TYPE_BURST          0x30    // Burst packet (binary, replaces text "BURST:")

// =============================================================================
// BINARY BURST PACKET - 8 bytes, sent during TX burst (OPT-1)
// =============================================================================
#define BINARY_BURST_SIZE       8

typedef struct __attribute__((packed)) {
    uint8_t  magic;             // 0xC5 (1)
    uint8_t  type;              // 0x30 = BURST (1)
    uint8_t  tx_mac[6];         // Transmitter MAC (6)
} binary_burst_t;               // Total: 8 bytes (vs ~40 bytes text)

_Static_assert(sizeof(binary_burst_t) == BINARY_BURST_SIZE,
               "Binary BURST packet must be exactly 8 bytes");

// Chip type (ESP32-C3 only)
#define CHIP_ESP32_C3           0x03

// Firmware version - v5.1.0-PURE (all backward compat removed)
#define FW_VERSION_MAJOR        5
#define FW_VERSION_MINOR        1
#define FW_VERSION_PATCH        0

// =============================================================================
// BINARY HELLO PACKET - 40 bytes, sent on boot and periodically
// =============================================================================
#define BINARY_HELLO_SIZE       40

typedef struct __attribute__((packed)) {
    uint8_t  magic;             // 0xC5 (1)
    uint8_t  type;              // 0x01 = HELLO (1)
    uint8_t  mac[6];            // Node MAC address (6)
    uint8_t  ip[4];             // IP address bytes (4)
    uint8_t  chip_type;         // CHIP_ESP32_C3 = 0x03 (1)
    uint8_t  fw_major;          // Firmware major (1)
    uint8_t  fw_minor;          // Firmware minor (1)
    uint8_t  fw_patch;          // Firmware patch (1)
    char     fw_tag[8];         // "BINARY\0\0" (8)
    uint32_t uptime_s;          // Uptime in seconds (4)
    uint32_t free_heap;         // Free heap bytes (4)
    uint32_t min_heap;          // Minimum heap seen (4)
    uint32_t crc32;             // CRC32 checksum (4)
} binary_hello_t;               // Total: 40 bytes

_Static_assert(sizeof(binary_hello_t) == BINARY_HELLO_SIZE,
               "Binary HELLO packet must be exactly 40 bytes");

// =============================================================================
// BINARY HEARTBEAT PACKET - 24 bytes, sent every 2 seconds
// =============================================================================
#define BINARY_HEARTBEAT_SIZE   24

typedef struct __attribute__((packed)) {
    uint8_t  magic;             // 0xC5 (1)
    uint8_t  type;              // 0x02 = HEARTBEAT (1)
    uint8_t  mac[6];            // Node MAC address (6)
    uint32_t uptime_s;          // Uptime in seconds (4)
    uint32_t free_heap;         // Free heap bytes (4)
    uint32_t min_heap;          // Minimum heap seen (4)
    uint32_t crc32;             // CRC32 checksum (4)
} binary_heartbeat_t;           // Total: 24 bytes

_Static_assert(sizeof(binary_heartbeat_t) == BINARY_HEARTBEAT_SIZE,
               "Binary HEARTBEAT packet must be exactly 24 bytes");

// =============================================================================
// BINARY TRIGGER PACKET - 32 bytes, received from server
// =============================================================================
#define BINARY_TRIGGER_SIZE     32

typedef struct __attribute__((packed)) {
    uint8_t  magic;             // 0xC5 (1)
    uint8_t  type;              // 0x10 = TRIGGER (1)
    uint8_t  tx_mac[6];         // MAC of transmitting node (6)
    uint32_t seq;               // Sequence number (4)
    uint16_t burst_count;       // Number of burst packets (2)
    uint16_t slot_ms;           // Slot duration in ms (2)
    uint8_t  samples_per_meas;  // Samples to collect (1)
    uint8_t  send_all_samples;  // 0=best only, 1=send all (1)
    uint8_t  reserved[10];      // Future use (10)
    uint32_t crc32;             // CRC32 checksum (4)
} binary_trigger_t;             // Total: 32 bytes

_Static_assert(sizeof(binary_trigger_t) == BINARY_TRIGGER_SIZE,
               "Binary TRIGGER packet must be exactly 32 bytes");

// =============================================================================
// BINARY DISCOVER PACKET - 8 bytes, received from server
// =============================================================================
#define BINARY_DISCOVER_SIZE    8

typedef struct __attribute__((packed)) {
    uint8_t  magic;             // 0xC5 (1)
    uint8_t  type;              // 0x11 = DISCOVER (1)
    uint16_t reserved;          // Padding (2)
    uint32_t crc32;             // CRC32 checksum (4)
} binary_discover_t;            // Total: 8 bytes

_Static_assert(sizeof(binary_discover_t) == BINARY_DISCOVER_SIZE,
               "Binary DISCOVER packet must be exactly 8 bytes");

// =============================================================================
// BINARY CSI BATCH V5.0 - Adaptive streaming with minimal headers
// =============================================================================
#define BINARY_CSI_MAGIC_V5_0   0xC5
#define BINARY_CSI_MAGIC_V5_1   0x07  // Version 5.0
#define BINARY_BATCH_HEADER_SIZE 8
#define BINARY_SAMPLE_SIZE_V5   107
#define MAX_SAMPLES_PER_BATCH   12    // (1400 MTU - 8 header - 4 CRC) / 107
#define BATCH_BUFFER_SIZE       1400  // MTU-sized buffer
#define SILENCE_TIMEOUT_US      3000  // 3ms silence = batch complete

// Batch header (8 bytes)
typedef struct __attribute__((packed)) {
    uint8_t  magic[2];          // 0xC5, 0x07 (2)
    uint8_t  slot_id;           // TX identity from trigger (1)
    uint8_t  sample_count;      // Number of samples in payload (1)
    uint16_t seq;               // Batch sequence number (2)
    uint16_t timestamp_lo;      // Low 16 bits of first sample timestamp (2)
} batch_header_v5_t;            // Total: 8 bytes

_Static_assert(sizeof(batch_header_v5_t) == BINARY_BATCH_HEADER_SIZE,
               "Batch header v5 must be exactly 8 bytes");

// Per-sample data (107 bytes)
typedef struct __attribute__((packed)) {
    uint16_t timestamp_delta;   // µs offset from batch timestamp (2)
    int8_t   rssi;              // Signal strength (1)
    int8_t   iq_data[104];      // 52 subcarriers × 2 (I/Q pairs) (104)
} sample_v5_t;                  // Total: 107 bytes

_Static_assert(sizeof(sample_v5_t) == BINARY_SAMPLE_SIZE_V5,
               "Sample v5 must be exactly 107 bytes");

// =============================================================================
// TRIGGER STATE STRUCTURE (for thread-safe access)
// =============================================================================
typedef struct {
    uint8_t tx_mac_bytes[6];    // Binary MAC for IRAM-safe comparison
    char    tx_mac_str[18];     // String MAC for logging
    uint32_t seq;               // Sequence number
    bool    active;             // Whether filtering is active
} trigger_state_t;

// =============================================================================
// GLOBAL STATE
// =============================================================================
static EventGroupHandle_t s_wifi_event_group = NULL;
static char node_mac[18] = {0};
static uint8_t node_mac_bytes[6] = {0};  // Binary MAC for efficient packing
static char node_ip[16] = {0};
static int udp_sock = -1;
static int trigger_sock = -1;
static int broadcast_sock = -1;  // Persistent broadcast socket
static int64_t boot_time_us = 0;
static atomic_bool is_transmitting = false;  // Atomic for cross-task visibility
static int wifi_retry_count = 0;
static uint32_t s_last_trigger_time_ms = 0;  // For trigger rate limiting

// Trigger state with spinlock protection for ISR-safe access
static trigger_state_t g_trigger_state = {0};
static portMUX_TYPE trigger_spinlock = portMUX_INITIALIZER_UNLOCKED;

// CSI ring buffer for reliable data capture
static RingbufHandle_t csi_ringbuf = NULL;

// Socket mutex for thread-safe UDP sends
static SemaphoreHandle_t socket_mutex = NULL;

// Server address (cached)
static struct sockaddr_in server_addr;
static struct sockaddr_in broadcast_addr;

// =============================================================================
// BATCH CSI STATE
// =============================================================================
static uint8_t g_batch_buffer[BATCH_BUFFER_SIZE];
static size_t g_batch_offset = BINARY_BATCH_HEADER_SIZE;  // Start after header
static uint8_t g_current_slot_id = 0;
static uint16_t g_batch_seq = 0;
static uint32_t g_batch_first_timestamp = 0;
static uint8_t g_batch_sample_count = 0;
static esp_timer_handle_t silence_timer = NULL;
static portMUX_TYPE batch_spinlock = portMUX_INITIALIZER_UNLOCKED;

// =============================================================================
// FORWARD DECLARATIONS
// =============================================================================
static void send_hello(void);

// =============================================================================
// IRAM FUNCTIONS - Time-critical code
// =============================================================================

/**
 * IRAM-safe 6-byte MAC comparison
 * Returns true if MACs match
 */
static inline bool IRAM_ATTR mac_bytes_equal(const uint8_t *mac1, const uint8_t *mac2)
{
    return (mac1[0] == mac2[0] && mac1[1] == mac2[1] && mac1[2] == mac2[2] &&
            mac1[3] == mac2[3] && mac1[4] == mac2[4] && mac1[5] == mac2[5]);
}

/**
 * CSI callback - runs in WiFi task context
 * IRAM_ATTR ensures this runs from RAM for lowest latency
 * Enhanced to capture AGC gain, noise floor, and signal mode for accuracy
 *
 * THREAD SAFETY:
 * - Uses spinlock-protected local copy of trigger state
 * - Uses atomic_load for is_transmitting flag
 * - All comparisons use binary data (no string functions)
 */
static void IRAM_ATTR csi_callback(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || atomic_load(&is_transmitting)) {
        return;
    }

    // Take local copy of trigger state under spinlock (ISR-safe)
    trigger_state_t local_state;
    portENTER_CRITICAL_ISR(&trigger_spinlock);
    local_state = g_trigger_state;
    portEXIT_CRITICAL_ISR(&trigger_spinlock);

    // Binary MAC comparison (IRAM-safe, no string functions)
    if (local_state.active) {
        if (!mac_bytes_equal(info->mac, local_state.tx_mac_bytes)) {
            return;
        }
    }

    // Build packet for ring buffer - capture all available metadata
    csi_packet_t pkt;
    pkt.timestamp_us = esp_timer_get_time();
    pkt.rssi = info->rx_ctrl.rssi;
    pkt.noise_floor = info->rx_ctrl.noise_floor;
    pkt.channel = info->rx_ctrl.channel;
    pkt.secondary_channel = info->rx_ctrl.secondary_channel;
    pkt.rate = info->rx_ctrl.rate;
    pkt.sig_mode = info->rx_ctrl.sig_mode;
    pkt.cwb = info->rx_ctrl.cwb;
    pkt.aggregation = info->rx_ctrl.aggregation;
    pkt.stbc = info->rx_ctrl.stbc;
    pkt.fec_coding = info->rx_ctrl.fec_coding;
    pkt.sgi = info->rx_ctrl.sgi;
    pkt.rx_state = 0;  // Critical for amplitude correction
    pkt.len = (info->len > MAX_CSI_DATA_LEN) ? MAX_CSI_DATA_LEN : info->len;
    memcpy(pkt.src_mac, info->mac, 6);
    memcpy(pkt.data, info->buf, pkt.len);

    // Send to ring buffer (non-blocking, ISR-safe with timeout=0)
    xRingbufferSend(csi_ringbuf, &pkt, sizeof(csi_packet_t), 0);
}

// =============================================================================
// WIFI EVENT HANDLER
// =============================================================================
static void wifi_event_handler(void *arg, esp_event_base_t event_base,
                               int32_t event_id, void *event_data)
{
    if (event_base == WIFI_EVENT) {
        switch (event_id) {
            case WIFI_EVENT_STA_START:
                // Don't auto-connect here - we connect manually after channel scan
                ESP_LOGI(TAG, "WiFi STA started");
                break;

            case WIFI_EVENT_STA_DISCONNECTED:
                wifi_retry_count++;
                xEventGroupClearBits(s_wifi_event_group, WIFI_CONNECTED_BIT);

                // Use exponential backoff: 1s, 2s, 4s, 5s (max), 5s, 5s...
                uint32_t delay_ms = (wifi_retry_count < 3) ? (1000 << wifi_retry_count) : WIFI_RECONNECT_DELAY_MS;
                if (delay_ms > WIFI_RECONNECT_DELAY_MS) delay_ms = WIFI_RECONNECT_DELAY_MS;

                ESP_LOGW(TAG, "WiFi disconnected, retry #%d in %lu ms",
                         wifi_retry_count, (unsigned long)delay_ms);

                // Signal failure only for initial connection (for boot sequence)
                if (wifi_retry_count >= WIFI_RETRY_MAX) {
                    xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
                }

                vTaskDelay(pdMS_TO_TICKS(delay_ms));
                esp_wifi_connect();  // Always keep trying - reliability is key
                break;

            case WIFI_EVENT_STA_CONNECTED:
                wifi_retry_count = 0;
                ESP_LOGI(TAG, "WiFi connected to AP");
                break;
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t *event = (ip_event_got_ip_t *)event_data;
        snprintf(node_ip, sizeof(node_ip), IPSTR, IP2STR(&event->ip_info.ip));
        ESP_LOGI(TAG, "Got IP: %s", node_ip);
        wifi_retry_count = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

// =============================================================================
// SCAN FOR AP ON PREFERRED CHANNEL
// =============================================================================
#if CONFIG_WIFI_CHANNEL > 0
static bool scan_for_ap_on_channel(uint8_t *out_bssid, uint8_t *out_channel)
{
    ESP_LOGI(TAG, "Scanning for SSID '%s' on channel %d...", CONFIG_WIFI_SSID, CONFIG_WIFI_CHANNEL);

    wifi_scan_config_t scan_config = {
        .ssid = (uint8_t *)CONFIG_WIFI_SSID,
        .bssid = NULL,
        .channel = 0,  // Scan all channels
        .show_hidden = false,
        .scan_type = WIFI_SCAN_TYPE_ACTIVE,
        .scan_time.active.min = 100,
        .scan_time.active.max = 300,
    };

    esp_err_t err = esp_wifi_scan_start(&scan_config, true);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "WiFi scan failed: %s", esp_err_to_name(err));
        return false;
    }

    uint16_t ap_count = 0;
    esp_wifi_scan_get_ap_num(&ap_count);

    if (ap_count == 0) {
        ESP_LOGW(TAG, "No APs found");
        return false;
    }

    wifi_ap_record_t *ap_records = malloc(sizeof(wifi_ap_record_t) * ap_count);
    if (!ap_records) {
        ESP_LOGE(TAG, "Failed to allocate memory for scan results");
        return false;
    }

    esp_wifi_scan_get_ap_records(&ap_count, ap_records);

    // Find AP on preferred channel, or best signal as fallback
    int preferred_idx = -1;
    int best_idx = -1;
    int8_t best_rssi = -127;

    for (int i = 0; i < ap_count; i++) {
        if (strcmp((char *)ap_records[i].ssid, CONFIG_WIFI_SSID) == 0) {
            ESP_LOGI(TAG, "  Found: ch=%d, rssi=%d, bssid=%02X:%02X:%02X:%02X:%02X:%02X",
                     ap_records[i].primary,
                     ap_records[i].rssi,
                     ap_records[i].bssid[0], ap_records[i].bssid[1], ap_records[i].bssid[2],
                     ap_records[i].bssid[3], ap_records[i].bssid[4], ap_records[i].bssid[5]);

            // Check if on preferred channel
            if (ap_records[i].primary == CONFIG_WIFI_CHANNEL) {
                if (preferred_idx < 0 || ap_records[i].rssi > ap_records[preferred_idx].rssi) {
                    preferred_idx = i;
                }
            }

            // Track best signal overall
            if (ap_records[i].rssi > best_rssi) {
                best_rssi = ap_records[i].rssi;
                best_idx = i;
            }
        }
    }

    int selected_idx = (preferred_idx >= 0) ? preferred_idx : best_idx;

    if (selected_idx >= 0) {
        memcpy(out_bssid, ap_records[selected_idx].bssid, 6);
        *out_channel = ap_records[selected_idx].primary;

        if (preferred_idx >= 0) {
            ESP_LOGI(TAG, "Selected AP on preferred channel %d", *out_channel);
        } else {
            ESP_LOGW(TAG, "No AP on channel %d, using best signal on channel %d",
                     CONFIG_WIFI_CHANNEL, *out_channel);
        }

        free(ap_records);
        return true;
    }

    ESP_LOGE(TAG, "No matching AP found for SSID '%s'", CONFIG_WIFI_SSID);
    free(ap_records);
    return false;
}
#endif

// =============================================================================
// INITIALIZE WIFI WITH CSI
// =============================================================================
static esp_err_t wifi_init_sta(void)
{
    s_wifi_event_group = xEventGroupCreate();
    if (!s_wifi_event_group) {
        ESP_LOGE(TAG, "Failed to create event group");
        return ESP_FAIL;
    }

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    // Increase WiFi task stack for CSI processing
    cfg.static_rx_buf_num = 16;
    cfg.dynamic_rx_buf_num = 64;
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    // Register event handlers
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, NULL));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(
        IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, NULL));

    wifi_config_t wifi_config = {
        .sta = {
            .ssid = CONFIG_WIFI_SSID,
            .password = CONFIG_WIFI_PASSWORD,
            .threshold.authmode = WIFI_AUTH_WPA2_PSK,
            .sae_pwe_h2e = WPA3_SAE_PWE_BOTH,
        },
    };

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));

    // Disable power save for lowest latency
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

    // Start WiFi first (required for scanning)
    ESP_ERROR_CHECK(esp_wifi_start());

    // Get MAC address (both string and binary forms)
    esp_wifi_get_mac(WIFI_IF_STA, node_mac_bytes);
    snprintf(node_mac, sizeof(node_mac), "%02X:%02X:%02X:%02X:%02X:%02X",
             node_mac_bytes[0], node_mac_bytes[1], node_mac_bytes[2],
             node_mac_bytes[3], node_mac_bytes[4], node_mac_bytes[5]);
    ESP_LOGI(TAG, "Node MAC: %s", node_mac);

#if CONFIG_WIFI_CHANNEL > 0
    // Scan for AP on preferred channel
    uint8_t target_bssid[6];
    uint8_t target_channel;

    if (scan_for_ap_on_channel(target_bssid, &target_channel)) {
        // Set BSSID to force connection to specific AP
        memcpy(wifi_config.sta.bssid, target_bssid, 6);
        wifi_config.sta.bssid_set = true;
        wifi_config.sta.channel = target_channel;

        ESP_LOGI(TAG, "Forcing connection to BSSID %02X:%02X:%02X:%02X:%02X:%02X on channel %d",
                 target_bssid[0], target_bssid[1], target_bssid[2],
                 target_bssid[3], target_bssid[4], target_bssid[5],
                 target_channel);
    } else {
        ESP_LOGW(TAG, "Channel scan failed, using default connection");
    }
#endif

    // Apply WiFi config and connect
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_connect());

    // Wait for connection with timeout
    ESP_LOGI(TAG, "Connecting to WiFi...");
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
            WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
            pdFALSE, pdFALSE, pdMS_TO_TICKS(30000));

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Connected to %s", CONFIG_WIFI_SSID);

        // Set bandwidth to 20MHz (required for CSI)
        ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));

        // Enable promiscuous mode for better CSI capture
        ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

        // Configure CSI
        wifi_csi_config_t csi_config = {
            .lltf_en = true,            // Legacy Long Training Field
            .htltf_en = true,           // HT-LTF
            .stbc_htltf2_en = true,     // STBC HT-LTF2
            .ltf_merge_en = true,       // Merge LTF
            .channel_filter_en = true,  // Enable channel filter
            .manu_scale = false,
            .shift = 0,                 // Auto scale (uint8_t, 0-15)
            .dump_ack_en = false,       // Don't dump ACK frames
        };

        ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
        ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(csi_callback, NULL));
        ESP_ERROR_CHECK(esp_wifi_set_csi(true));

        ESP_LOGI(TAG, "CSI capture enabled (HT20, promiscuous mode)");
        return ESP_OK;
    }

    ESP_LOGE(TAG, "WiFi connection timeout");
    return ESP_FAIL;
}

// =============================================================================
// SOCKET INITIALIZATION
// =============================================================================
static esp_err_t init_sockets(void)
{
    ESP_LOGI(TAG, "Initializing sockets...");
    ESP_LOGI(TAG, "  Server: %s:%d", CONFIG_SERVER_IP, CONFIG_SERVER_PORT);
    ESP_LOGI(TAG, "  Trigger port: %d", CONFIG_TRIGGER_PORT);
    ESP_LOGI(TAG, "  Burst port: %d", CONFIG_BURST_PORT);

    // Cache server address
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(CONFIG_SERVER_PORT);
    if (inet_pton(AF_INET, CONFIG_SERVER_IP, &server_addr.sin_addr) != 1) {
        ESP_LOGE(TAG, "Invalid server IP address: %s", CONFIG_SERVER_IP);
        return ESP_FAIL;
    }

    // Cache broadcast address
    memset(&broadcast_addr, 0, sizeof(broadcast_addr));
    broadcast_addr.sin_family = AF_INET;
    broadcast_addr.sin_port = htons(CONFIG_BURST_PORT);
    broadcast_addr.sin_addr.s_addr = htonl(INADDR_BROADCAST);

    // Create main UDP socket
    udp_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (udp_sock < 0) {
        ESP_LOGE(TAG, "Failed to create UDP socket: errno %d", errno);
        return ESP_FAIL;
    }

    // Set socket options
    int opt = 1;
    setsockopt(udp_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Create persistent broadcast socket
    broadcast_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (broadcast_sock < 0) {
        ESP_LOGE(TAG, "Failed to create broadcast socket: errno %d", errno);
        return ESP_FAIL;
    }

    int broadcast = 1;
    setsockopt(broadcast_sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
    setsockopt(broadcast_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Create trigger socket
    trigger_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (trigger_sock < 0) {
        ESP_LOGE(TAG, "Failed to create trigger socket: errno %d", errno);
        return ESP_FAIL;
    }

    setsockopt(trigger_sock, SOL_SOCKET, SO_BROADCAST, &broadcast, sizeof(broadcast));
    setsockopt(trigger_sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    struct sockaddr_in trigger_addr = {
        .sin_family = AF_INET,
        .sin_port = htons(CONFIG_TRIGGER_PORT),
        .sin_addr.s_addr = htonl(INADDR_ANY),
    };

    if (bind(trigger_sock, (struct sockaddr *)&trigger_addr, sizeof(trigger_addr)) < 0) {
        ESP_LOGE(TAG, "Failed to bind trigger socket: errno %d", errno);
        return ESP_FAIL;
    }

    // Set receive timeout on trigger socket to prevent blocking forever
    struct timeval tv = {
        .tv_sec = SOCKET_TIMEOUT_MS / 1000,
        .tv_usec = (SOCKET_TIMEOUT_MS % 1000) * 1000
    };
    setsockopt(trigger_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

    ESP_LOGI(TAG, "Sockets initialized (server=%s:%d, trigger=%d, burst=%d)",
             CONFIG_SERVER_IP, CONFIG_SERVER_PORT, CONFIG_TRIGGER_PORT, CONFIG_BURST_PORT);
    return ESP_OK;
}

// =============================================================================
// UDP SEND HELPERS (Thread-safe with mutex)
// =============================================================================
static inline void send_to_server(const char *data, size_t len)
{
    if (udp_sock >= 0 && socket_mutex) {
        if (xSemaphoreTake(socket_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            sendto(udp_sock, data, len, 0,
                   (struct sockaddr *)&server_addr, sizeof(server_addr));
            xSemaphoreGive(socket_mutex);
        }
    }
}

static inline void send_broadcast_packet(const char *data, size_t len)
{
    if (broadcast_sock >= 0 && socket_mutex) {
        if (xSemaphoreTake(socket_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            sendto(broadcast_sock, data, len, 0,
                   (struct sockaddr *)&broadcast_addr, sizeof(broadcast_addr));
            xSemaphoreGive(socket_mutex);
        }
    }
}

// =============================================================================
// HELPER: Parse IP string "192.168.1.100" to 4 bytes
// =============================================================================
static void ip_str_to_bytes(const char *ip_str, uint8_t *ip_bytes)
{
    unsigned int a, b, c, d;
    if (sscanf(ip_str, "%u.%u.%u.%u", &a, &b, &c, &d) == 4) {
        ip_bytes[0] = (uint8_t)a;
        ip_bytes[1] = (uint8_t)b;
        ip_bytes[2] = (uint8_t)c;
        ip_bytes[3] = (uint8_t)d;
    } else {
        memset(ip_bytes, 0, 4);
    }
}

// =============================================================================
// PACKET BUILDERS - Binary protocol (zero malloc)
// =============================================================================

/**
 * Send HELLO packet - 40 bytes, no malloc
 * Used for node registration and periodic re-announcement
 */
static void send_hello(void)
{
    binary_hello_t pkt = {0};  // Zero-initialize

    // Header
    pkt.magic = BINARY_PROTO_MAGIC;
    pkt.type = MSG_TYPE_HELLO;

    // Node identification
    memcpy(pkt.mac, node_mac_bytes, 6);
    ip_str_to_bytes(node_ip, pkt.ip);

    // Chip and firmware info
    pkt.chip_type = CHIP_ESP32_C3;
    pkt.fw_major = FW_VERSION_MAJOR;
    pkt.fw_minor = FW_VERSION_MINOR;
    pkt.fw_patch = FW_VERSION_PATCH;
    strncpy(pkt.fw_tag, "PURE", sizeof(pkt.fw_tag) - 1);

    // Runtime stats
    pkt.uptime_s = (uint32_t)((esp_timer_get_time() - boot_time_us) / 1000000);
    pkt.free_heap = esp_get_free_heap_size();
    pkt.min_heap = esp_get_minimum_free_heap_size();

    // CRC32 (over packet excluding CRC field itself)
    pkt.crc32 = esp_crc32_le(0, (const uint8_t *)&pkt,
                              sizeof(binary_hello_t) - sizeof(uint32_t));

    send_to_server((const char *)&pkt, sizeof(binary_hello_t));
    ESP_LOGD(TAG, "HELLO sent (40 bytes, binary)");
}

/**
 * Send HEARTBEAT packet - 24 bytes, no malloc
 * Used for keepalive and health monitoring
 */
static void send_heartbeat(void)
{
    binary_heartbeat_t pkt = {0};  // Zero-initialize

    // Header
    pkt.magic = BINARY_PROTO_MAGIC;
    pkt.type = MSG_TYPE_HEARTBEAT;

    // Node identification
    memcpy(pkt.mac, node_mac_bytes, 6);

    // Runtime stats
    pkt.uptime_s = (uint32_t)((esp_timer_get_time() - boot_time_us) / 1000000);
    pkt.free_heap = esp_get_free_heap_size();
    pkt.min_heap = esp_get_minimum_free_heap_size();

    // CRC32
    pkt.crc32 = esp_crc32_le(0, (const uint8_t *)&pkt,
                              sizeof(binary_heartbeat_t) - sizeof(uint32_t));

    send_to_server((const char *)&pkt, sizeof(binary_heartbeat_t));
}

// OPT-4: Inline macro for pilot phase extraction (eliminates function call overhead)
// Pilot subcarriers are at indices 6, 20, 32, 46 in the 52-subcarrier array
#define EXTRACT_PILOT_PHASES(iq, pilots) do { \
    (pilots)[0] = (iq)[PILOT_SC_1 * 2];         \
    (pilots)[1] = (iq)[PILOT_SC_1 * 2 + 1];     \
    (pilots)[2] = (iq)[PILOT_SC_2 * 2];         \
    (pilots)[3] = (iq)[PILOT_SC_2 * 2 + 1];     \
    (pilots)[4] = (iq)[PILOT_SC_3 * 2];         \
    (pilots)[5] = (iq)[PILOT_SC_3 * 2 + 1];     \
    (pilots)[6] = (iq)[PILOT_SC_4 * 2];         \
    (pilots)[7] = (iq)[PILOT_SC_4 * 2 + 1];     \
} while(0)

// =============================================================================
// BATCH CSI FUNCTIONS
// =============================================================================

/**
 * Finalize and send the current batch
 * Called from silence timer or when batch is full
 */
static void finalize_and_send_batch_v5(void)
{
    portENTER_CRITICAL(&batch_spinlock);

    if (g_batch_sample_count == 0) {
        portEXIT_CRITICAL(&batch_spinlock);
        return;
    }

    // Build header
    batch_header_v5_t *header = (batch_header_v5_t *)g_batch_buffer;
    header->magic[0] = BINARY_CSI_MAGIC_V5_0;
    header->magic[1] = BINARY_CSI_MAGIC_V5_1;
    header->slot_id = g_current_slot_id;
    header->sample_count = g_batch_sample_count;
    header->seq = g_batch_seq++;
    header->timestamp_lo = (uint16_t)(g_batch_first_timestamp & 0xFFFF);

    size_t data_size = g_batch_offset;
    uint8_t sample_count = g_batch_sample_count;

    // Compute and append CRC32 (over header + samples)
    uint32_t crc = esp_crc32_le(0, g_batch_buffer, data_size);
    memcpy(g_batch_buffer + data_size, &crc, 4);
    size_t total_size = data_size + 4;  // Include CRC trailer

    // Reset for next batch
    g_batch_offset = BINARY_BATCH_HEADER_SIZE;
    g_batch_sample_count = 0;
    g_batch_first_timestamp = 0;

    portEXIT_CRITICAL(&batch_spinlock);

    // Send outside critical section
    send_to_server((const char *)g_batch_buffer, total_size);

    ESP_LOGD(TAG, "v5.0 batch sent: slot=%d, samples=%d, size=%d",
             g_current_slot_id, sample_count, total_size);
}

/**
 * Silence timer callback - flush batch after 3ms of no CSI
 */
static void silence_timer_callback(void *arg)
{
    finalize_and_send_batch_v5();
}

/**
 * Initialize v5.0 batch mode and silence timer
 */
static esp_err_t init_batch_mode_v5(void)
{
    esp_timer_create_args_t timer_args = {
        .callback = silence_timer_callback,
        .arg = NULL,
        .dispatch_method = ESP_TIMER_TASK,
        .name = "silence_timer"
    };

    esp_err_t ret = esp_timer_create(&timer_args, &silence_timer);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to create silence timer: %s", esp_err_to_name(ret));
        return ret;
    }

    ESP_LOGI(TAG, "Protocol v5.0 batch mode initialized (timeout=%dµs)", SILENCE_TIMEOUT_US);
    return ESP_OK;
}

/**
 * Start new batch for a TDMA slot
 */
static void start_new_batch_v5(uint8_t slot_id)
{
    // Flush any pending batch first
    finalize_and_send_batch_v5();

    portENTER_CRITICAL(&batch_spinlock);
    g_current_slot_id = slot_id;
    g_batch_offset = BINARY_BATCH_HEADER_SIZE;
    g_batch_sample_count = 0;
    g_batch_first_timestamp = 0;
    portEXIT_CRITICAL(&batch_spinlock);

    ESP_LOGD(TAG, "v5.0 new batch started: slot=%d", slot_id);
}

/**
 * Add CSI sample to current batch
 * Returns true if sample added, false if batch is full (caller should flush)
 */
static bool add_sample_to_batch_v5(const csi_packet_t *pkt)
{
    portENTER_CRITICAL(&batch_spinlock);

    // Check if batch is full
    if (g_batch_sample_count >= MAX_SAMPLES_PER_BATCH ||
        (g_batch_offset + BINARY_SAMPLE_SIZE_V5) > BATCH_BUFFER_SIZE) {
        portEXIT_CRITICAL(&batch_spinlock);
        finalize_and_send_batch_v5();

        // Try again after flush
        portENTER_CRITICAL(&batch_spinlock);
    }

    // Record first timestamp
    if (g_batch_sample_count == 0) {
        g_batch_first_timestamp = (uint32_t)(pkt->timestamp_us & 0xFFFFFFFF);
    }

    // Build sample at current offset
    sample_v5_t *sample = (sample_v5_t *)(g_batch_buffer + g_batch_offset);
    sample->timestamp_delta = (uint16_t)((pkt->timestamp_us - g_batch_first_timestamp) & 0xFFFF);
    sample->rssi = pkt->rssi;

    // Copy I/Q data (max 104 bytes = 52 subcarriers × 2)
    size_t iq_bytes = (pkt->len < 104) ? pkt->len : 104;
    memcpy(sample->iq_data, pkt->data, iq_bytes);
    if (iq_bytes < 104) {
        memset(sample->iq_data + iq_bytes, 0, 104 - iq_bytes);
    }

    g_batch_offset += BINARY_SAMPLE_SIZE_V5;
    g_batch_sample_count++;

    portEXIT_CRITICAL(&batch_spinlock);

    // Restart silence timer
    esp_timer_stop(silence_timer);
    esp_timer_start_once(silence_timer, SILENCE_TIMEOUT_US);

    return true;
}

// =============================================================================
// TRANSMIT BURST - Precise timing with configurable delay
// =============================================================================
static void transmit_burst(uint32_t count, uint32_t seq)
{
    (void)seq;  // seq not needed in binary burst (8 bytes vs 40 bytes)
    atomic_store(&is_transmitting, true);

    // Use configured burst count if 0 is passed
    if (count == 0) {
        count = CONFIG_BURST_COUNT;
    }

    // Pre-build binary burst packet (only 8 bytes vs ~40 bytes text)
    binary_burst_t pkt = {
        .magic = BINARY_PROTO_MAGIC,
        .type = MSG_TYPE_BURST,
    };
    memcpy(pkt.tx_mac, node_mac_bytes, 6);

    ESP_LOGD(TAG, "Starting burst TX: %" PRIu32 " packets (binary), %d us delay",
             count, CONFIG_BURST_DELAY_US);

    for (uint32_t i = 0; i < count; i++) {
        send_broadcast_packet((const char *)&pkt, sizeof(binary_burst_t));

        // Precise inter-packet delay for consistent timing
        if (CONFIG_BURST_DELAY_US > 0) {
            esp_rom_delay_us(CONFIG_BURST_DELAY_US);
        }

        // Yield periodically to prevent watchdog timeout
        if ((i & 0x1F) == 0x1F) {  // Every 32 packets
            vTaskDelay(1);
        }
    }

    atomic_store(&is_transmitting, false);
    ESP_LOGD(TAG, "Burst TX complete: %" PRIu32 " packets", count);
}

// =============================================================================
// HELPER: Validate binary packet CRC32
// =============================================================================
static bool validate_crc32(const uint8_t *data, size_t len)
{
    if (len < 4) return false;

    // CRC32 is last 4 bytes
    uint32_t received_crc;
    memcpy(&received_crc, data + len - 4, sizeof(uint32_t));

    // Calculate CRC over packet excluding CRC field
    uint32_t calculated_crc = esp_crc32_le(0, data, len - 4);

    return (received_crc == calculated_crc);
}

// =============================================================================
// HELPER: Convert MAC bytes to string for logging
// =============================================================================
static void mac_bytes_to_str(const uint8_t *mac_bytes, char *mac_str)
{
    snprintf(mac_str, 18, "%02X:%02X:%02X:%02X:%02X:%02X",
             mac_bytes[0], mac_bytes[1], mac_bytes[2],
             mac_bytes[3], mac_bytes[4], mac_bytes[5]);
}

// =============================================================================
// HELPER: Process trigger and collect CSI
// =============================================================================
static void process_trigger(const uint8_t *tx_mac_bytes, uint32_t seq,
                           uint16_t burst_count, uint16_t slot_ms,
                           uint8_t slot_id)
{
    // Build MAC string for logging and CSI packet
    char tx_mac_str[18];
    mac_bytes_to_str(tx_mac_bytes, tx_mac_str);

    // Build new trigger state
    trigger_state_t new_state = {0};
    memcpy(new_state.tx_mac_bytes, tx_mac_bytes, 6);
    strncpy(new_state.tx_mac_str, tx_mac_str, sizeof(new_state.tx_mac_str) - 1);
    new_state.seq = seq;
    new_state.active = true;

    // Update global trigger state atomically
    portENTER_CRITICAL(&trigger_spinlock);
    g_trigger_state = new_state;
    portEXIT_CRITICAL(&trigger_spinlock);

    // Check if we are the transmitter
    if (mac_bytes_equal(tx_mac_bytes, node_mac_bytes)) {
        ESP_LOGD(TAG, "TRIGGER: TX mode, seq=%" PRIu32 ", bursts=%u", seq, burst_count);
        transmit_burst(burst_count, seq);
        return;
    }

    // We are receiver - collect CSI samples
    ESP_LOGD(TAG, "TRIGGER: RX mode, collecting CSI from %s", tx_mac_str);

    // Clear any old data from ring buffer
    size_t item_size;
    void *item;
    while ((item = xRingbufferReceive(csi_ringbuf, &item_size, 0)) != NULL) {
        vRingbufferReturnItem(csi_ringbuf, item);
    }

    // Wait for CSI data
    vTaskDelay(pdMS_TO_TICKS(slot_ms));

    // Batch all CSI samples into single packet
    start_new_batch_v5(slot_id);

    uint8_t sample_count = 0;
    while ((item = xRingbufferReceive(csi_ringbuf, &item_size, 0)) != NULL) {
        csi_packet_t *pkt = (csi_packet_t *)item;
        add_sample_to_batch_v5(pkt);
        sample_count++;
        vRingbufferReturnItem(csi_ringbuf, item);
    }

    // Flush the batch
    finalize_and_send_batch_v5();

    if (sample_count > 0) {
        ESP_LOGD(TAG, "Batch sent: slot=%d, samples=%d", slot_id, sample_count);
    } else {
        ESP_LOGD(TAG, "No CSI data received");
    }

    // Clear trigger state
    portENTER_CRITICAL(&trigger_spinlock);
    g_trigger_state.active = false;
    portEXIT_CRITICAL(&trigger_spinlock);
}

// =============================================================================
// TRIGGER HANDLER TASK - Binary protocol with JSON fallback
// =============================================================================
static void trigger_task(void *pvParameters)
{
    ESP_LOGI(TAG, "Trigger listener started on port %d (binary protocol)", CONFIG_TRIGGER_PORT);

    // Subscribe to task watchdog
    esp_task_wdt_add(NULL);

    uint8_t rx_buffer[512];
    struct sockaddr_in source_addr;
    socklen_t socklen = sizeof(source_addr);
    uint32_t loop_count = 0;

    while (1) {
        // Feed watchdog every loop iteration
        esp_task_wdt_reset();

        // Periodic heap check
        if ((++loop_count % 50) == 0) {
            uint32_t free_heap = esp_get_free_heap_size();
            uint32_t min_heap = esp_get_minimum_free_heap_size();

            if (free_heap < HEAP_CRITICAL_THRESHOLD) {
                ESP_LOGE(TAG, "CRITICAL: Heap exhausted (%lu bytes), rebooting!",
                         (unsigned long)free_heap);
                vTaskDelay(pdMS_TO_TICKS(100));
                esp_restart();
            } else if (free_heap < HEAP_WARNING_THRESHOLD) {
                ESP_LOGW(TAG, "WARNING: Low heap: %lu bytes (min: %lu)",
                         (unsigned long)free_heap, (unsigned long)min_heap);
            }
        }

        int len = recvfrom(trigger_sock, rx_buffer, sizeof(rx_buffer) - 1, 0,
                          (struct sockaddr *)&source_addr, &socklen);

        if (len < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                continue;  // Normal timeout
            }
            ESP_LOGE(TAG, "recvfrom failed: errno %d", errno);
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        // Check for binary protocol (magic byte 0xC5)
        if (len >= 2 && rx_buffer[0] == BINARY_PROTO_MAGIC) {
            uint8_t msg_type = rx_buffer[1];

            switch (msg_type) {
                case MSG_TYPE_DISCOVER:
                    if (len >= BINARY_DISCOVER_SIZE && validate_crc32(rx_buffer, len)) {
                        ESP_LOGI(TAG, "DISCOVER received (binary)");
                        send_hello();
                    } else {
                        ESP_LOGW(TAG, "Invalid DISCOVER packet (len=%d, crc=%s)",
                                 len, validate_crc32(rx_buffer, len) ? "ok" : "fail");
                    }
                    break;

                case MSG_TYPE_TRIGGER:
                    if (len >= BINARY_TRIGGER_SIZE && validate_crc32(rx_buffer, len)) {
                        binary_trigger_t *trig = (binary_trigger_t *)rx_buffer;

                        // Rate limiting: ignore triggers that arrive too fast
                        uint32_t now_ms = xTaskGetTickCount() * portTICK_PERIOD_MS;
                        uint32_t elapsed_ms = now_ms - s_last_trigger_time_ms;

                        if (elapsed_ms < MIN_TRIGGER_INTERVAL_MS) {
                            ESP_LOGD(TAG, "Trigger skipped: only %" PRIu32 "ms since last (min %dms)",
                                     elapsed_ms, MIN_TRIGGER_INTERVAL_MS);
                            break;  // Skip this trigger, wait for next
                        }

                        s_last_trigger_time_ms = now_ms;

                        // Simplified logging for production
                        ESP_LOGD(TAG, "TRIGGER seq=%" PRIu32, trig->seq);

                        process_trigger(
                            trig->tx_mac,
                            trig->seq,
                            trig->burst_count ? trig->burst_count : CONFIG_BURST_COUNT,
                            trig->slot_ms ? trig->slot_ms : 80,
                            trig->reserved[0]  // slot_id
                        );
                    } else {
                        ESP_LOGW(TAG, "Invalid TRIGGER packet (len=%d, crc=%s)",
                                 len, validate_crc32(rx_buffer, len) ? "ok" : "fail");
                    }
                    break;

                default:
                    ESP_LOGD(TAG, "Unknown binary message type: 0x%02X", msg_type);
                    break;
            }
            continue;
        }

        // Non-binary packet - ignore silently (JSON support removed in v4.1)
    }
}

// =============================================================================
// HEARTBEAT TASK (with watchdog and WiFi monitoring)
// =============================================================================
static void heartbeat_task(void *pvParameters)
{
    // Subscribe to task watchdog
    esp_task_wdt_add(NULL);

    // Initial announcements with delay
    vTaskDelay(pdMS_TO_TICKS(1000));
    for (int i = 0; i < 3; i++) {
        esp_task_wdt_reset();
        send_hello();
        vTaskDelay(pdMS_TO_TICKS(500));
    }

    TickType_t last_hello = xTaskGetTickCount();
    TickType_t last_heartbeat = xTaskGetTickCount();

    while (1) {
        // Feed watchdog
        esp_task_wdt_reset();

        TickType_t now = xTaskGetTickCount();

        // Check WiFi connection status
        EventBits_t bits = xEventGroupGetBits(s_wifi_event_group);
        bool wifi_connected = (bits & WIFI_CONNECTED_BIT) != 0;

        // Send HELLO periodically for re-discovery (only if connected)
        if (wifi_connected && (now - last_hello) >= pdMS_TO_TICKS(ANNOUNCE_INTERVAL_MS)) {
            send_hello();
            last_hello = now;
        }

        // Send heartbeat (only if connected)
        if (wifi_connected && (now - last_heartbeat) >= pdMS_TO_TICKS(HEARTBEAT_INTERVAL_MS)) {
            send_heartbeat();
            last_heartbeat = now;
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }
}

// =============================================================================
// DIAGNOSTIC FUNCTIONS
// =============================================================================
static void print_startup_config(void)
{
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  WiFi Tomography Node v%d.%d.%d-PURE",
             FW_VERSION_MAJOR, FW_VERSION_MINOR, FW_VERSION_PATCH);
    ESP_LOGI(TAG, "  ESP32-C3 - Batch CSI Protocol");
    ESP_LOGI(TAG, "  Built: %s %s", __DATE__, __TIME__);
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "Configuration:");
    ESP_LOGI(TAG, "  WiFi SSID:     %s", CONFIG_WIFI_SSID);
    ESP_LOGI(TAG, "  WiFi Channel:  %d %s", CONFIG_WIFI_CHANNEL,
             CONFIG_WIFI_CHANNEL > 0 ? "(forced)" : "(auto)");
    ESP_LOGI(TAG, "  Server IP:     %s", CONFIG_SERVER_IP);
    ESP_LOGI(TAG, "  Server Port:   %d", CONFIG_SERVER_PORT);
    ESP_LOGI(TAG, "  Trigger Port:  %d", CONFIG_TRIGGER_PORT);
    ESP_LOGI(TAG, "  Burst Port:    %d", CONFIG_BURST_PORT);
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "CSI Settings:");
    ESP_LOGI(TAG, "  Burst count:   %d packets", CONFIG_BURST_COUNT);
    ESP_LOGI(TAG, "  Burst delay:   %d us", CONFIG_BURST_DELAY_US);
    ESP_LOGI(TAG, "  Protocol:      Batch (CRC32 validated)");
    ESP_LOGI(TAG, "========================================");
}

static void test_server_connectivity(void)
{
    ESP_LOGI(TAG, "Testing server connectivity...");

    // Create a simple test socket
    int test_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (test_sock < 0) {
        ESP_LOGE(TAG, "  FAIL: Cannot create test socket (errno %d)", errno);
        return;
    }

    // Set short timeout
    struct timeval tv = { .tv_sec = 2, .tv_usec = 0 };
    setsockopt(test_sock, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));

    // Binary test packet (2 bytes: magic + type)
    uint8_t test_pkt[2] = { BINARY_PROTO_MAGIC, MSG_TYPE_TEST };
    int sent = sendto(test_sock, test_pkt, sizeof(test_pkt), 0,
                      (struct sockaddr *)&server_addr, sizeof(server_addr));

    if (sent > 0) {
        ESP_LOGI(TAG, "  OK: Sent %d bytes to %s:%d", sent,
                 CONFIG_SERVER_IP, CONFIG_SERVER_PORT);
    } else {
        ESP_LOGE(TAG, "  FAIL: sendto failed (errno %d)", errno);
    }

    close(test_sock);
}

// =============================================================================
// MAIN APPLICATION
// =============================================================================
void app_main(void)
{
    print_startup_config();

    boot_time_us = esp_timer_get_time();

    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_LOGW(TAG, "NVS flash erase required");
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Create CSI ring buffer
    csi_ringbuf = xRingbufferCreate(CSI_RINGBUF_SIZE, RINGBUF_TYPE_NOSPLIT);
    if (!csi_ringbuf) {
        ESP_LOGE(TAG, "Failed to create CSI ring buffer");
        return;
    }

    // Create socket mutex for thread-safe UDP sends
    socket_mutex = xSemaphoreCreateMutex();
    if (!socket_mutex) {
        ESP_LOGE(TAG, "Failed to create socket mutex");
        return;
    }

    // Initialize WiFi with CSI
    if (wifi_init_sta() != ESP_OK) {
        ESP_LOGE(TAG, "WiFi initialization failed");
        return;
    }

    // Initialize sockets
    if (init_sockets() != ESP_OK) {
        ESP_LOGE(TAG, "Socket initialization failed");
        return;
    }

    // Test server connectivity
    test_server_connectivity();

    // Initialize batch mode
    if (init_batch_mode_v5() != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize batch mode");
        return;
    }
    ESP_LOGI(TAG, "Batch CSI mode enabled");

    // Initialize task watchdog for reliability
    // This will automatically reboot if tasks hang for >30 seconds
    esp_task_wdt_config_t wdt_config = {
        .timeout_ms = WATCHDOG_TIMEOUT_S * 1000,
        .idle_core_mask = 0,  // Don't watch idle tasks
        .trigger_panic = true,  // Reboot on timeout
    };
    esp_task_wdt_reconfigure(&wdt_config);
    ESP_LOGI(TAG, "Task watchdog configured: %d second timeout", WATCHDOG_TIMEOUT_S);

    // Start tasks with appropriate priorities and stack sizes
    xTaskCreate(trigger_task, "trigger", 4096, NULL, 5, NULL);
    xTaskCreate(heartbeat_task, "heartbeat", 2048, NULL, 3, NULL);  // Reduced from 4096 - heartbeat is simple

    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "  NODE READY");
    ESP_LOGI(TAG, "  MAC: %s", node_mac);
    ESP_LOGI(TAG, "  IP:  %s", node_ip);
    ESP_LOGI(TAG, "  Heap: %" PRIu32 " bytes free", esp_get_free_heap_size());
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "Sending initial HELLO messages...");
}
