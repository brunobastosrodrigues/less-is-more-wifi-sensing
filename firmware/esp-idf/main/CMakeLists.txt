idf_component_register(SRCS "main.c"
                    INCLUDE_DIRS "."
                    REQUIRES
                        nvs_flash
                        esp_wifi
                        esp_event
                        esp_netif
                        log
                        lwip
                        esp_ringbuf
                        esp_timer)
# Note: json (cJSON) removed in v4.1 - pure binary protocol

# Suppress stringop-truncation warning (ESP-IDF v5.5.2 + GCC 14 compatibility)
target_compile_options(__idf_main PRIVATE -Wno-error=stringop-truncation -Wno-error=unused-function)