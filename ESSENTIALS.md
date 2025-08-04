# Neonatal Jaundice Detection System - Essential Guide

This document provides a clear summary of the essential components and instructions for the Neonatal Jaundice Detection System.

## What's Working Now

The jaundice detection system is now working on your Raspberry Pi with:

- Continuous monitoring using the USB camera
- Automatic detection and alerting
- Saving images when jaundice is detected
- Automatic service startup on boot

## Essential Files

These are the only files you need to keep and maintain:

1. **`raspberry_pi_optimized.py`** - The main monitoring script
2. **`jaundice_mobilenetv3.onnx`** - The machine learning model
3. **`setup_raspberry_pi.sh`** - Setup script for new installations
4. **`fix_camera_complete.sh`** - Troubleshooting script (keep for emergencies)

## Files That Can Be Removed

The following scripts were created during development and troubleshooting and are no longer needed:

- `simple_monitor.py`
- `raspberry_monitor.py`
- `raspberry_monitor_fixed.py`
- `camera_test_minimal.py`
- `troubleshoot.sh`
- `simple_install.sh`
- `fix_dependencies.sh`
- `fix_camera.sh` (replaced by fix_camera_complete.sh)
- `emergency_camera_fix.sh`
- `venv_troubleshoot.sh`

The `prev_scripts` folder can also be removed as it contains outdated versions.

## Quick Start Guide

### On Raspberry Pi

1. **Start the service**:

   ```bash
   sudo systemctl start jaundice_monitor.service
   ```

2. **Check status**:

   ```bash
   sudo systemctl status jaundice_monitor.service
   ```

3. **View detection logs**:

   ```bash
   tail -f ~/jaundice_monitor/jaundice_detection_log.txt
   ```

4. **View saved images**:
   ```bash
   ls -la ~/jaundice_monitor/detections/
   ```

### Fresh Installation

If setting up on a new Raspberry Pi:

1. **Create project directory**:

   ```bash
   mkdir -p ~/jaundice_monitor
   ```

2. **Copy essential files**:

   ```bash
   # Copy these files to the jaundice_monitor directory:
   # - raspberry_pi_optimized.py
   # - jaundice_mobilenetv3.onnx
   # - setup_raspberry_pi.sh
   # - fix_camera_complete.sh
   ```

3. **Run setup script**:
   ```bash
   cd ~/jaundice_monitor
   chmod +x setup_raspberry_pi.sh
   ./setup_raspberry_pi.sh
   ```

## Troubleshooting

If the camera stops working:

1. **Run the comprehensive fix script**:

   ```bash
   cd ~/jaundice_monitor
   chmod +x fix_camera_complete.sh
   ./fix_camera_complete.sh
   ```

2. **Restart the service**:
   ```bash
   sudo systemctl restart jaundice_monitor.service
   ```

## Technical Summary

- **Model**: MobileNetV3-Small converted to ONNX (5.81 MB)
- **Camera**: USB webcam connected to Raspberry Pi
- **Detection rate**: Approximately 5 frames per second on Pi 3B+
- **Storage usage**: Low (only saves images when jaundice detected)
- **Camera access methods**: OpenCV with automatic fallback to fswebcam

## Notes

- The system uses a virtual environment with system site packages
- The monitoring script automatically switches between camera methods if one fails
- Service auto-restarts if it crashes
- Service auto-starts on boot
- Detection images are saved in ~/jaundice_monitor/detections/
