#!/usr/bin/env python3
"""
Extremely simple camera test script for Raspberry Pi
Uses multiple methods to try to access camera
"""

import time
import os
import sys
import subprocess
import datetime

print("==== CAMERA TEST SCRIPT ====")
print(f"Date/Time: {datetime.datetime.now()}")
print(f"Python version: {sys.version}")

# Try to import OpenCV
print("\n=== Testing OpenCV Import ===")
try:
    import cv2
    print(f"OpenCV version: {cv2.__version__}")
except ImportError as e:
    print(f"Failed to import OpenCV: {e}")
    print("Installing OpenCV...")
    try:
        subprocess.run(["pip", "install", "opencv-python-headless"], check=True)
        import cv2
        print(f"OpenCV installed and imported: {cv2.__version__}")
    except Exception as e:
        print(f"Failed to install OpenCV: {e}")

# Try numpy import
print("\n=== Testing NumPy Import ===")
try:
    import numpy as np
    print(f"NumPy version: {np.__version__}")
except ImportError as e:
    print(f"Failed to import NumPy: {e}")

# Print device info
print("\n=== System Information ===")
try:
    cpu_info = subprocess.check_output("cat /proc/cpuinfo | grep 'model name' | head -1", shell=True).decode().strip()
    print(cpu_info)
except:
    print("Could not get CPU info")

try:
    mem_info = subprocess.check_output("free -h | head -2", shell=True).decode().strip()
    print(mem_info)
except:
    print("Could not get memory info")

# List all video devices
print("\n=== Available Video Devices ===")
video_devices = []
try:
    if os.path.exists("/dev"):
        devices = [d for d in os.listdir("/dev") if d.startswith("video")]
        if devices:
            print(f"Found devices: {devices}")
            video_devices = [f"/dev/{d}" for d in devices]
        else:
            print("No video devices found in /dev")
    else:
        print("/dev directory not found")
except Exception as e:
    print(f"Error listing devices: {e}")

# Try to open each camera with OpenCV
if 'cv2' in sys.modules:
    print("\n=== Testing OpenCV Camera Access ===")
    for i in range(5):  # Try indices 0-4
        print(f"\nTesting camera index {i}...")
        try:
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                print(f"Failed to open camera at index {i}")
            else:
                print(f"Successfully opened camera at index {i}")
                # Try to read a frame
                ret, frame = cap.read()
                if ret:
                    print(f"Successfully read frame: shape={frame.shape}")
                    # Save the frame
                    filename = f"opencv_test_{i}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved test image to {filename}")
                else:
                    print("Failed to read frame")
                cap.release()
        except Exception as e:
            print(f"Error testing camera {i}: {e}")

# Try to capture with fswebcam
print("\n=== Testing fswebcam ===")
try:
    for device in video_devices:
        print(f"\nTesting {device} with fswebcam...")
        try:
            output = subprocess.check_output(
                ["fswebcam", "-d", device, "-r", "320x240", "--no-banner", f"fswebcam_test_{os.path.basename(device)}.jpg"],
                stderr=subprocess.STDOUT
            ).decode()
            print(output)
            print(f"Capture attempt complete for {device}")
        except subprocess.CalledProcessError as e:
            print(f"fswebcam error for {device}: {e.output.decode() if e.output else str(e)}")
except Exception as e:
    print(f"Error with fswebcam: {e}")

# Attempt with libcamera (for Pi camera)
print("\n=== Testing libcamera-still ===")
try:
    output = subprocess.check_output(
        ["libcamera-still", "-o", "libcamera_test.jpg", "--immediate"],
        stderr=subprocess.STDOUT
    ).decode()
    print(output)
    print("libcamera-still completed")
except subprocess.CalledProcessError as e:
    print(f"libcamera error: {e.output.decode() if e.output else str(e)}")
except FileNotFoundError:
    print("libcamera-still not found (normal if not using Pi camera module)")

print("\n==== TEST COMPLETE ====")
print("If any images were captured, they are saved in the current directory")
print("Check for files like opencv_test_*.jpg or fswebcam_test_*.jpg")

# List captured images
print("\n=== Captured Images ===")
try:
    captured_images = [f for f in os.listdir(".") if f.endswith(".jpg") and 
                     ("test" in f or "opencv" in f or "fswebcam" in f or "libcamera" in f)]
    if captured_images:
        for img in captured_images:
            print(f"- {img}")
    else:
        print("No captured images found")
except Exception as e:
    print(f"Error listing images: {e}")
