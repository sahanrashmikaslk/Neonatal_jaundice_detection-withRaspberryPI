#!/bin/bash
# Camera troubleshooting script for Raspberry Pi

echo "==== USB Camera Troubleshooting ===="

# Check for available cameras
echo "1. Checking for available video devices..."
if ls /dev/video* 2>/dev/null; then
    echo "✓ Found video devices"
else
    echo "✗ No video devices found!"
    echo "   - Ensure camera is properly connected"
    echo "   - Try reconnecting the USB camera"
fi

# Check user groups
echo ""
echo "2. Checking user permissions..."
if groups sahan | grep -q video; then
    echo "✓ User 'sahan' is in the video group"
else
    echo "✗ User 'sahan' is NOT in the video group!"
    echo "   - Add user to video group: sudo usermod -a -G video sahan"
    echo "   - Log out and log back in for changes to take effect"
fi

# Check kernel modules
echo ""
echo "3. Checking for required kernel modules..."
echo "V4L2 driver status:"
if lsmod | grep -q videodev; then
    echo "✓ Video4Linux (V4L2) driver is loaded"
else
    echo "✗ V4L2 driver not loaded!"
    echo "   - Try loading it: sudo modprobe videodev"
fi

# Check camera info with v4l2-ctl
echo ""
echo "4. Detailed camera information:"
if command -v v4l2-ctl &> /dev/null; then
    for device in /dev/video*; do
        echo "Device: $device"
        v4l2-ctl --device=$device --info
        v4l2-ctl --device=$device --list-formats-ext
    done
else
    echo "v4l2-ctl not installed. Install with: sudo apt install v4l-utils"
fi

# Attempt to capture a still image with fswebcam (more reliable than OpenCV for some cameras)
echo ""
echo "5. Testing camera with fswebcam (alternative method)..."
if ! command -v fswebcam &> /dev/null; then
    echo "fswebcam not installed. Installing now..."
    sudo apt install -y fswebcam
fi

echo "Capturing test image with fswebcam..."
fswebcam -r 640x480 --no-banner ~/jaundice_monitor/fswebcam_test.jpg
if [ -f ~/jaundice_monitor/fswebcam_test.jpg ]; then
    echo "✓ Successfully captured image with fswebcam"
    echo "   Image saved to: ~/jaundice_monitor/fswebcam_test.jpg"
else
    echo "✗ Failed to capture image with fswebcam"
fi

# Restart udev to refresh USB devices
echo ""
echo "6. Restarting udev to refresh USB devices..."
sudo udevadm control --reload-rules
sudo udevadm trigger

# Test camera directly with OpenCV
echo ""
echo "7. Testing camera with basic OpenCV script..."
cat > ~/jaundice_monitor/basic_camera_test.py << 'EOF'
#!/usr/bin/env python3
import cv2
import time
import sys

print("OpenCV version:", cv2.__version__)
print("Testing basic camera access...")

for camera_index in range(4):  # Test camera indices 0-3
    print(f"Trying camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"  Failed to open camera at index {camera_index}")
    else:
        print(f"  Successfully opened camera at index {camera_index}")
        print(f"  Camera properties:")
        print(f"  - Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  - Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        # Try to capture a frame
        ret, frame = cap.read()
        if ret:
            print(f"  Successfully captured frame: {frame.shape}")
            cv2.imwrite(f"camera_{camera_index}_test.jpg", frame)
            print(f"  Saved test image to camera_{camera_index}_test.jpg")
        else:
            print(f"  Failed to capture frame")
        
        cap.release()
        print("")

print("Camera testing complete!")
EOF

chmod +x ~/jaundice_monitor/basic_camera_test.py
echo "Running basic camera test..."
cd ~/jaundice_monitor
source ~/jaundice_monitor/venv/bin/activate
python ~/jaundice_monitor/basic_camera_test.py

echo ""
echo "==== Troubleshooting Summary ===="
echo "If no cameras were detected, try the following:"
echo "1. Reboot the Raspberry Pi: sudo reboot"
echo "2. Try a different USB port"
echo "3. Try setting a different camera index in the service file"
echo "4. Check if your camera works on another computer"
echo "5. Try a different USB camera"
echo ""
echo "To modify the service to use a different camera index:"
echo "  sudo nano /etc/systemd/system/jaundice_monitor.service"
echo "  Change the --camera parameter to a different number (0, 1, 2, etc.)"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl restart jaundice_monitor.service"
