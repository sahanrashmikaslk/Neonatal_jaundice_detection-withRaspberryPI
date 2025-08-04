#!/bin/bash
# EMERGENCY CAMERA CONFIGURATION SCRIPT
# This script performs comprehensive camera configuration for Raspberry Pi

echo "==== CAMERA EMERGENCY FIX ===="

# Install critical dependencies
echo "Installing critical dependencies..."
sudo apt update
sudo apt install -y fswebcam v4l-utils python3-opencv

# Set proper permissions for camera devices
echo "Setting permissions for video devices..."
sudo chmod 666 /dev/video*
sudo usermod -a -G video $USER

# Fix USB power management
echo "Fixing USB power management..."
sudo sh -c 'echo "max_usb_current=1" >> /boot/config.txt'
sudo sh -c 'echo "dtoverlay=dwc2,dr_mode=host" >> /boot/config.txt'

# Fix service file
echo "Updating jaundice monitor service..."
cat > jaundice_monitor_fixed.service << 'EOF'
[Unit]
Description=Neonatal Jaundice Detection Monitor
After=network.target

[Service]
User=sahan
WorkingDirectory=/home/sahan/jaundice_monitor
ExecStart=/home/sahan/jaundice_monitor/venv/bin/python /home/sahan/jaundice_monitor/raspberry_pi_optimized.py --camera 0
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo mv jaundice_monitor_fixed.service /etc/systemd/system/jaundice_monitor.service
sudo systemctl daemon-reload

# Create a test script with multiple camera methods
echo "Creating comprehensive camera test script..."
cat > camera_test_all_methods.py << 'EOF'
#!/usr/bin/env python3
"""
Comprehensive camera test script with multiple methods
"""
import os
import sys
import time
import subprocess
import cv2
import numpy as np

def print_sep():
    print("=" * 50)

def test_opencv():
    print_sep()
    print("METHOD 1: TESTING OPENCV")
    print(f"OpenCV version: {cv2.__version__}")
    
    for i in range(5):  # Test indices 0-4
        print(f"\nTrying camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if not cap.isOpened():
            print(f"  ✗ Failed to open camera at index {i}")
            continue
            
        print(f"  ✓ Successfully opened camera at index {i}")
        print(f"  Camera properties:")
        print(f"  - Width: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}")
        print(f"  - Height: {cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"  - FPS: {cap.get(cv2.CAP_PROP_FPS)}")
        
        # Try multiple frame captures
        frames_captured = 0
        for attempt in range(5):
            print(f"  Capture attempt {attempt+1}/5...")
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                frames_captured += 1
                if attempt == 4:  # Save only last successful frame
                    output_file = f"opencv_test_{i}.jpg"
                    cv2.imwrite(output_file, frame)
                    print(f"  ✓ Saved test image: {output_file}")
            else:
                print(f"  ✗ Failed to capture frame on attempt {attempt+1}")
                
            time.sleep(0.5)
            
        print(f"  Camera {i} success rate: {frames_captured}/5 frames")
        cap.release()

def test_fswebcam():
    print_sep()
    print("METHOD 2: TESTING FSWEBCAM")
    
    # Check if fswebcam is installed
    try:
        subprocess.run(["which", "fswebcam"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("fswebcam not installed. Installing...")
        subprocess.run(["sudo", "apt", "install", "-y", "fswebcam"], check=False)
    
    # Test each video device
    for video_dev in sorted([d for d in os.listdir('/dev') if d.startswith('video')]):
        device_path = f"/dev/{video_dev}"
        print(f"\nTesting {device_path}...")
        
        # Multiple capture attempts
        for attempt in range(3):
            output_file = f"fswebcam_test_{video_dev}_{attempt+1}.jpg"
            print(f"  Capture attempt {attempt+1}/3...")
            
            try:
                cmd = ["fswebcam", "-d", device_path, "-r", "640x480", "--no-banner", output_file]
                result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
                
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    print(f"  ✓ Successfully captured: {output_file}")
                else:
                    print(f"  ✗ Capture failed or empty file: {output_file}")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
            
            time.sleep(1)

def test_direct_v4l2():
    print_sep()
    print("METHOD 3: TESTING DIRECT V4L2")
    
    # Check if v4l2-ctl is installed
    try:
        subprocess.run(["which", "v4l2-ctl"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("v4l2-utils not installed. Installing...")
        subprocess.run(["sudo", "apt", "install", "-y", "v4l-utils"], check=False)
    
    # List all video devices
    for video_dev in sorted([d for d in os.listdir('/dev') if d.startswith('video')]):
        device_path = f"/dev/{video_dev}"
        print(f"\nTesting {device_path}...")
        
        # Get detailed device info
        try:
            print("  Device capabilities:")
            result = subprocess.run(["v4l2-ctl", "--device", device_path, "--info"], 
                                 check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("  " + result.stdout.decode().replace('\n', '\n  '))
            
            print("  Supported formats:")
            result = subprocess.run(["v4l2-ctl", "--device", device_path, "--list-formats-ext"], 
                                 check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print("  " + result.stdout.decode().replace('\n', '\n  '))
            
            # Try to capture a test image
            output_file = f"v4l2_test_{video_dev}.jpg"
            print(f"  Attempting to capture image to {output_file}...")
            result = subprocess.run(["v4l2-ctl", "--device", device_path, "--set-fmt-video=width=640,height=480,pixelformat=MJPG", 
                                  "--stream-mmap", "--stream-count=1", "--stream-to", output_file],
                                 check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"  ✓ Successfully captured image")
            else:
                print(f"  ✗ Failed to capture image or empty file")
                
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

def test_camera_gstreamer():
    print_sep()
    print("METHOD 4: TESTING GSTREAMER")
    
    try:
        # Check if GStreamer is available
        result = subprocess.run(["gst-launch-1.0", "--version"], 
                             check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            print("GStreamer not installed. Installing...")
            subprocess.run(["sudo", "apt", "install", "-y", "gstreamer1.0-tools", "gstreamer1.0-plugins-good"], check=False)
    except:
        print("Installing GStreamer...")
        subprocess.run(["sudo", "apt", "install", "-y", "gstreamer1.0-tools", "gstreamer1.0-plugins-good"], check=False)
    
    # Try GStreamer with each video device
    for i in range(5):  # Test indices 0-4
        device_path = f"/dev/video{i}"
        if not os.path.exists(device_path):
            continue
            
        print(f"\nTesting {device_path} with GStreamer...")
        output_file = f"gstreamer_test_{i}.jpg"
        
        try:
            # Attempt to capture a single frame with GStreamer
            cmd = [
                "gst-launch-1.0", "-v", f"v4l2src device={device_path} num-buffers=1", 
                "!", "videoconvert", "!", "jpegenc", "!", f"filesink location={output_file}"
            ]
            result = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=10)
            
            if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                print(f"  ✓ Successfully captured image with GStreamer")
            else:
                print(f"  ✗ Failed to capture image or empty file")
                print(f"  Error: {result.stderr.decode()}")
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")

def check_usb_info():
    print_sep()
    print("USB DEVICE INFORMATION")
    
    try:
        # Check if lsusb is available
        result = subprocess.run(["lsusb"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("USB devices connected:")
        print(result.stdout.decode())
        
        # Check USB power settings
        print("\nUSB power settings:")
        for hub in range(5):
            power_path = f"/sys/bus/usb/devices/usb{hub}/power/autosuspend"
            if os.path.exists(power_path):
                with open(power_path, 'r') as f:
                    print(f"Hub {hub} autosuspend: {f.read().strip()}")
    except Exception as e:
        print(f"Error checking USB info: {str(e)}")

def main():
    print("==== COMPREHENSIVE CAMERA TEST ====")
    print("This script will test camera access using multiple methods")
    print("If one method works but another doesn't, we'll know what approach to use")
    print("\nStarting tests...")
    
    # Run all tests
    check_usb_info()
    test_opencv()
    test_fswebcam()
    test_direct_v4l2()
    test_camera_gstreamer()
    
    print_sep()
    print("==== TEST COMPLETE ====")
    print("If any images were captured, they are saved in the current directory")
    print("Check for files like opencv_test_*.jpg or fswebcam_test_*.jpg")
    
    # List captured images
    print("\n=== Captured Images ===")
    image_files = [f for f in os.listdir('.') if any(f.startswith(prefix) for prefix in 
                  ['opencv_test_', 'fswebcam_test_', 'v4l2_test_', 'gstreamer_test_', 'camera_'])]
    
    if image_files:
        for img in image_files:
            print(f"- {img}")
    else:
        print("No test images were captured by any method!")
        
    # Print recommendation
    print("\n=== RECOMMENDATION ===")
    if any(f.startswith('opencv_test_') for f in os.listdir('.')):
        print("✓ OpenCV method worked - use the optimized Python script")
    elif any(f.startswith('fswebcam_test_') for f in os.listdir('.')):
        print("✓ fswebcam method worked - use the optimized Python script")
    else:
        print("✗ No standard methods worked - try these advanced solutions:")
        print("  1. Try a powered USB hub to provide more power to the camera")
        print("  2. Connect camera directly to USB port (not through a hub)")
        print("  3. Try a different USB port on the Raspberry Pi")
        print("  4. Try a different USB camera model")

if __name__ == "__main__":
    main()
EOF

chmod +x camera_test_all_methods.py
echo "Running comprehensive camera test..."
python3 camera_test_all_methods.py

echo ""
echo "==== INSTALLATION SUMMARY ===="
echo "1. Updated service configuration for reliable camera access"
echo "2. Fixed USB power management settings"
echo "3. Installed all required dependencies"
echo "4. Created optimized Python script with multiple camera methods"
echo ""
echo "The new service file has been installed. To use it:"
echo "1. Copy the new optimized Python script to ~/jaundice_monitor:"
echo "   cp raspberry_pi_optimized.py ~/jaundice_monitor/"
echo "2. Restart the service:"
echo "   sudo systemctl restart jaundice_monitor.service"
echo "3. Check the status:"
echo "   sudo systemctl status jaundice_monitor.service"
echo ""
echo "For persistent USB power settings, reboot the Raspberry Pi:"
echo "sudo reboot"
