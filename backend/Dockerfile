# =============================================================================
# ESP32 WiFi CSI TDMA System - Backend Docker Image
# =============================================================================
#
# This Dockerfile builds a minimal Python container for the TDMA scheduler
# and CSI receiver. It's designed for reliability over features.
#
# Usage:
#   docker build -t esp32-csi-tdma .
#   docker run --network host esp32-csi-tdma
#
# IMPORTANT: --network host is required for UDP communication with ESP32 nodes
# =============================================================================

FROM python:3.10-slim

# Prevent Python from buffering stdout (important for Docker logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python source code
COPY src/ .

# Create data directory for CSI storage
RUN mkdir -p /app/csi_data

# Expose UDP port for CSI data (5000) and trigger broadcasts (5001)
# Note: With --network host, these are just documentation
EXPOSE 5000/udp
EXPOSE 5001/udp

# Health check - verify Python can import main modules
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import config; import scheduler; import receiver" || exit 1

# Run the server
CMD ["python", "main.py"]
