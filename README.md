# Neonatal Jaundice Detection System for Raspberry Pi

A continuous monitoring system for detecting neonatal jaundice using machine learning, designed specifically for Raspberry Pi deployment in incubator settings.

<!-- ![Live Feed Detection](./ScreenShots/LiveFeedDetection.png) -->

## Project Overview

This project implements a deep learning model to detect potential signs of neonatal jaundice from images of an infant's skin or eyes. The system runs as a headless monitoring service on Raspberry Pi, making it ideal for continuous infant monitoring in an incubator environment.

## Model Details

- **Architecture:** MobileNetV3-Small (optimized for edge devices)
- **Framework:** ONNX Runtime (optimized for Raspberry Pi)
- **Training Data:** [Kaggle Jaundice Image Data](https://www.kaggle.com/datasets/aiolapo/jaundice-image-data)
  - Approx. 200 Jaundiced images
  - Approx. 560 Normal images
- **Input Size:** 224x224 pixels (RGB)
- **Output:** Binary classification (Normal/Jaundice) with probability score

## Key Features

- **Continuous Monitoring System:**

  - Headless operation for 24/7 infant monitoring
  - Optimized ONNX model for better performance on Raspberry Pi
  - Multiple camera access methods for reliability
  - Automatic alert logging and image saving
  - Service-based operation with auto-restart capability

- **Robust Camera Handling:**

  - Automatic switching between OpenCV and fswebcam methods
  - Recovery from camera disconnections
  - Support for various USB webcams
  - Built-in error handling and diagnostics

- **Optimized for Raspberry Pi:**
  - Low resource usage suitable for Raspberry Pi 3B+
  - Efficient frame processing
  - Minimal dependencies

## Essential Files

- `raspberry_pi_optimized.py` - Main monitoring script
- `jaundice_mobilenetv3.onnx` - ONNX model file
- `setup_raspberry_pi.sh` - Setup script
- `fix_camera_complete.sh` - Camera troubleshooting script

## Installation on Raspberry Pi

1. **Transfer essential files to the Raspberry Pi**:

   - `raspberry_pi_optimized.py`
   - `jaundice_mobilenetv3.onnx`
   - `setup_raspberry_pi.sh`
   - `fix_camera_complete.sh`

2. **Create project directory and setup**:

   ```bash
   mkdir -p ~/jaundice_monitor
   cd ~/jaundice_monitor
   # Copy the files to this directory
   chmod +x setup_raspberry_pi.sh
   ./setup_raspberry_pi.sh
   ```

3. **Check service status**:

   ```bash
   sudo systemctl status jaundice_monitor.service
   ```

4. **View detection logs**:
   ```bash
   tail -f ~/jaundice_monitor/jaundice_detection_log.txt
   ```

## System Operation

The monitoring system runs as a service that:

1. **Continuously monitors**: Captures and analyzes frames automatically
2. **Logs detections**: Records all jaundice detections with timestamp and confidence
3. **Saves images**: Stores images of positive detections in the `detections` folder
4. **Auto-recovers**: Automatically handles camera disconnections and errors

## Service Management

Control the monitoring service with these commands:

```bash
# Start the service
sudo systemctl start jaundice_monitor.service

# Stop the service
sudo systemctl stop jaundice_monitor.service

# Check service status
sudo systemctl status jaundice_monitor.service

# Enable auto-start on boot
sudo systemctl enable jaundice_monitor.service

# View logs
journalctl -u jaundice_monitor.service
```

## Troubleshooting

If camera connection fails:

1. **Run the camera fix script**:

   ```bash
   cd ~/jaundice_monitor
   chmod +x fix_camera_complete.sh
   ./fix_camera_complete.sh
   ```

2. **Check USB power issues**:

   - Try a powered USB hub for better camera stability
   - Connect camera directly to Raspberry Pi USB port
   - Raspberry Pi 3B+ has limited USB power which can affect camera operation

3. **Camera hardware checks**:

   - Verify camera works on another computer
   - Try different USB ports on the Raspberry Pi
   - Check if camera is recognized with `lsusb` command

4. **Service diagnostics**:

   ```bash
   # Check detailed service logs
   journalctl -u jaundice_monitor.service

   # Check camera detection
   ls /dev/video*
   ```

## Project Structure

```
.
├── raspberry_pi_optimized.py   # Main Raspberry Pi monitoring script
├── jaundice_mobilenetv3.onnx   # ONNX model file for deployment
├── setup_raspberry_pi.sh       # Setup script
├── fix_camera_complete.sh      # Camera troubleshooting script
└── ESSENTIALS.md               # Guide to essential files
```

## Technical Details

### Model Information

- **Model Size**: 5.81 MB (ONNX format)
- **Base Model**: MobileNetV3-Small
- **Preprocessing**: Resize to 224x224, normalize with ImageNet stats
- **Inference Speed**: ~5 FPS on Raspberry Pi 3B+
- **Accuracy**: Over 90% on test dataset

### Raspberry Pi Optimizations

- **Multiple Camera Methods**: OpenCV with fswebcam fallback
- **Frame Buffer Management**: Optimized to reduce memory usage
- **Thread Control**: Limited to 2 threads for better Pi performance
- **Error Handling**: Robust recovery from camera disconnections
- **Service Management**: Systemd service with auto-restart

### Hardware Requirements

- **Recommended**: Raspberry Pi 3B+ or newer
- **Storage**: At least 1GB free space
- **Camera**: USB webcam with UVC support
- **Power**: Recommended to use official Raspberry Pi power supply
- **Optional**: Powered USB hub for better camera reliability

## Command-line Arguments

The monitoring script supports several command-line arguments:

```bash
python raspberry_pi_optimized.py [OPTIONS]

Options:
  --camera INTEGER      Camera index (default: 0)
  --display             Display video feed (not recommended for headless)
  --threshold FLOAT     Alert threshold for jaundice probability (default: 0.7)
  --single              Single detection mode (exit after one detection)
```

## Additional Resources

- For a complete guide to the essential files, see [ESSENTIALS.md](./ESSENTIALS.md)
- Original dataset: [Kaggle Jaundice Image Data](https://www.kaggle.com/datasets/aiolapo/jaundice-image-data)

## License

[Your license information]

## Acknowledgments

- Dataset provided by [Kaggle Jaundice Image Data](https://www.kaggle.com/datasets/aiolapo/jaundice-image-data)
- MobileNetV3 architecture by Howard et al.
