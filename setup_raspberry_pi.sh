#!/bin/bash
# Setup script for Neonatal Jaundice Detection on Raspberry Pi
# This script helps set up the environment and install required packages

echo "==== Setting up Neonatal Jaundice Detection on Raspberry Pi ===="
echo "This script will install required packages and set up the monitoring service."

# Create directories
echo "Creating project directory..."
mkdir -p ~/jaundice_monitor
cd ~/jaundice_monitor

# Update and install dependencies
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

echo "Installing required packages..."
sudo apt install -y python3-pip python3-opencv libopenjp2-7 libtiff5 libatlas-base-dev

# Install Python packages
echo "Installing Python dependencies (this may take a while)..."
pip3 install numpy opencv-python-headless onnxruntime

# Ask user for camera configuration
echo ""
echo "USB camera configuration:"
echo "1) Default camera (usually /dev/video0)"
echo "2) Specify camera device"
read -p "Select option (1/2): " camera_option

CAMERA_INDEX=0
if [ "$camera_option" = "2" ]; then
    echo "Available video devices:"
    ls -l /dev/video*
    read -p "Enter camera index (number only, e.g., for /dev/video2 enter '2'): " cam_idx
    CAMERA_INDEX=$cam_idx
fi

# Create a test script to verify camera
echo "Creating camera test script..."
cat > ~/jaundice_monitor/test_camera.py << 'EOF'
import cv2
import sys
import time

camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
print(f"Attempting to open camera at index {camera_index}")

cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"Failed to open camera at index {camera_index}")
    sys.exit(1)

print("Camera opened successfully. Capturing 5 frames...")
for i in range(5):
    ret, frame = cap.read()
    if ret:
        print(f"Frame {i+1}: Shape={frame.shape}")
        # Save one test frame
        if i == 2:
            cv2.imwrite('test_frame.jpg', frame)
            print("Saved test frame as 'test_frame.jpg'")
    else:
        print(f"Failed to capture frame {i+1}")
    time.sleep(1)

cap.release()
print("Camera test complete!")
EOF

echo "Testing camera..."
python3 ~/jaundice_monitor/test_camera.py $CAMERA_INDEX

# Modify service file with the correct camera index
echo "Configuring service file..."
cat > ~/jaundice_monitor/jaundice_monitor.service << EOF
[Unit]
Description=Neonatal Jaundice Monitoring Service
After=network.target

[Service]
ExecStart=/usr/bin/python3 /home/sahan/jaundice_monitor/raspberry_monitor.py --camera $CAMERA_INDEX
WorkingDirectory=/home/sahan/jaundice_monitor
StandardOutput=append:/home/sahan/jaundice_monitor/service.log
StandardError=append:/home/sahan/jaundice_monitor/service.log
Restart=always
User=sahan

[Install]
WantedBy=multi-user.target
EOF

# Install service
echo "Installing service..."
sudo cp ~/jaundice_monitor/jaundice_monitor.service /etc/systemd/system/
sudo systemctl daemon-reload

echo ""
echo "====== Setup Complete ======"
echo "To start monitoring, transfer your model and monitoring script, then run:"
echo "  sudo systemctl enable jaundice_monitor.service"
echo "  sudo systemctl start jaundice_monitor.service"
echo ""
echo "To check status:"
echo "  sudo systemctl status jaundice_monitor.service"
echo ""
echo "To view logs:"
echo "  tail -f ~/jaundice_monitor/jaundice_detection_log.txt"
echo "  tail -f ~/jaundice_monitor/service.log"
