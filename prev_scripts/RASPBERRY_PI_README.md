# Neonatal Jaundice Monitoring for Raspberry Pi

This guide explains how to set up and run the neonatal jaundice detection system on a Raspberry Pi 3B+ with a USB webcam for continuous monitoring in an incubator setting.

## Hardware Requirements

- Raspberry Pi 3B+ with Raspberry OS 64-bit Lite
- USB webcam (compatible with Linux/V4L2)
- Power supply for Raspberry Pi
- SD card (16GB+ recommended)
- Optional: case for Raspberry Pi

## Setup Instructions

### 1. Prepare the Files

First, convert your PyTorch model to ONNX format on your development machine:

```bash
# On your development machine
python convert_to_onnx.py
```

This will create the `jaundice_mobilenetv3.onnx` file that's optimized for inference on the Raspberry Pi.

### 2. Transfer Files to Raspberry Pi

Transfer the following files to your Raspberry Pi using SCP:

```bash
# From your development machine
scp raspberry_monitor.py sahan@192.168.8.137:~/
scp jaundice_mobilenetv3.onnx sahan@192.168.8.137:~/
scp setup_raspberry_pi.sh sahan@192.168.8.137:~/
```

### 3. Run the Setup Script

SSH into your Raspberry Pi and run the setup script:

```bash
ssh sahan@192.168.8.137
chmod +x setup_raspberry_pi.sh
./setup_raspberry_pi.sh
```

The script will:

- Install required packages
- Set up the project directory
- Test your USB camera
- Configure the service file

### 4. Move Files to Project Directory

After running the setup script, move your files to the project directory:

```bash
mv raspberry_monitor.py ~/jaundice_monitor/
mv jaundice_mobilenetv3.onnx ~/jaundice_monitor/
```

### 5. Start the Monitoring Service

Enable and start the service:

```bash
sudo systemctl enable jaundice_monitor.service
sudo systemctl start jaundice_monitor.service
```

### 6. Monitor the System

Check the status of the service:

```bash
sudo systemctl status jaundice_monitor.service
```

View the detection logs:

```bash
tail -f ~/jaundice_monitor/jaundice_detection_log.txt
```

## Usage

The system will continuously monitor using the USB webcam and log any jaundice detections. When jaundice is detected with high probability (above the threshold), the system will:

1. Log the detection with timestamp
2. Save the image (at configured intervals to avoid filling storage)

## Troubleshooting

### Camera Issues

If the camera isn't detected:

1. Check camera connection:

   ```bash
   ls -l /dev/video*
   ```

2. Test camera directly:
   ```bash
   python3 ~/jaundice_monitor/test_camera.py 0  # Replace 0 with camera index if needed
   ```

### Service Issues

If the service fails to start:

1. Check service logs:

   ```bash
   sudo journalctl -u jaundice_monitor.service
   ```

2. Check service status:

   ```bash
   sudo systemctl status jaundice_monitor.service
   ```

3. Try running the script manually to see any errors:
   ```bash
   cd ~/jaundice_monitor
   python3 raspberry_monitor.py --camera 0
   ```

## Advanced Configuration

You can modify the behavior by editing the `raspberry_monitor.py` file:

- Adjust `SAVE_INTERVAL` to change how often images are saved
- Modify `alert_threshold` when starting the service to change detection sensitivity
- Add email or other notification mechanisms in the alert section

## Performance Notes

The Raspberry Pi 3B+ has limited computational resources. The script is optimized by:

1. Using ONNX Runtime instead of PyTorch for inference
2. Reducing the camera resolution
3. Optimizing the preprocessing pipeline
4. Using efficient thread management

For better performance, consider:

- Using a Raspberry Pi 4 if available
- Ensuring good cooling for the Raspberry Pi
- Closing other unnecessary services
