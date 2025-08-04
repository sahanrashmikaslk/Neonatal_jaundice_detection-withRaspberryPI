#!/bin/bash
# Simple install script for Raspberry Pi Jaundice Detection
# This script creates a virtual environment for Python packages

echo "==== Installing Jaundice Detection Dependencies ===="

# Update system
echo "Updating system packages..."
sudo apt update
sudo apt install -y python3-pip python3-venv python3-dev libatlas-base-dev python3-full
sudo apt install -y libopenjp2-7 libtiff5 libavcodec-dev libavformat-dev libswscale-dev

# Install OpenCV system dependencies
echo "Installing OpenCV dependencies..."
sudo apt install -y libopencv-dev python3-opencv

# Create virtual environment with system site packages
echo "Creating Python virtual environment (with access to system packages)..."
mkdir -p ~/jaundice_monitor/venv
python3 -m venv ~/jaundice_monitor/venv --system-site-packages

# Activate virtual environment
source ~/jaundice_monitor/venv/bin/activate

# Install dependencies in virtual environment
echo "Installing NumPy and ONNX Runtime in virtual environment..."
pip install numpy
pip install onnxruntime

# Install OpenCV in virtual environment if system package not accessible
echo "Installing OpenCV in virtual environment (if needed)..."
python -c "import cv2" 2>/dev/null || pip install opencv-python-headless

# Verify installations
echo ""
echo "Verifying installations (in virtual environment):"
echo "OpenCV: "
python -c "import cv2; print('Success! OpenCV version:', cv2.__version__)" || echo "Failed to import cv2"

echo "NumPy: "
python -c "import numpy; print('Success! NumPy version:', numpy.__version__)" || echo "Failed to import numpy"

echo "ONNX Runtime: "
python -c "import onnxruntime; print('Success! ONNX Runtime version:', onnxruntime.__version__)" || echo "Failed to import onnxruntime"

# Setting permissions for scripts and ensuring camera access
echo ""
echo "Setting permissions for scripts and camera..."
chmod +x ~/jaundice_monitor/raspberry_monitor.py

# Ensure user has access to camera
echo "Ensuring camera access..."
sudo usermod -a -G video sahan

# Create a camera test script
echo "Creating camera test script..."
cat > ~/jaundice_monitor/test_camera.py << 'EOF'
#!/usr/bin/env python3
import cv2
import time
import sys

def test_camera(camera_index=0):
    print(f"Testing camera at index {camera_index}")
    print("Opening camera...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"ERROR: Failed to open camera at index {camera_index}")
        return False
    
    print("Camera opened successfully!")
    print("Capturing test frames...")
    
    for i in range(3):
        ret, frame = cap.read()
        if ret:
            print(f"Frame {i+1}: Shape={frame.shape}")
            if i == 1:  # Save the middle frame
                filename = f"camera_test_{camera_index}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved test image to {filename}")
        else:
            print(f"Failed to capture frame {i+1}")
        time.sleep(1)
    
    cap.release()
    print("Camera test complete!")
    return True

if __name__ == "__main__":
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    success = test_camera(camera_index)
    sys.exit(0 if success else 1)
EOF

chmod +x ~/jaundice_monitor/test_camera.py

# Run camera test from within the virtual environment
echo "Testing camera (this will help verify camera access)..."
python ~/jaundice_monitor/test_camera.py 0

# Update service file to use virtual environment
echo "Updating service file to use virtual environment..."
cat > ~/jaundice_monitor/jaundice_monitor.service << EOF
[Unit]
Description=Neonatal Jaundice Monitoring Service
After=network.target

[Service]
ExecStart=/home/sahan/jaundice_monitor/venv/bin/python /home/sahan/jaundice_monitor/raspberry_monitor.py --camera 0
WorkingDirectory=/home/sahan/jaundice_monitor
StandardOutput=append:/home/sahan/jaundice_monitor/service.log
StandardError=append:/home/sahan/jaundice_monitor/service.log
Restart=always
User=sahan

[Install]
WantedBy=multi-user.target
EOF

# Install updated service
sudo cp ~/jaundice_monitor/jaundice_monitor.service /etc/systemd/system/
sudo systemctl daemon-reload

# Create a model test script
echo "Creating model test script..."
cat > ~/jaundice_monitor/test_model.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import numpy as np
import onnxruntime as ort

MODEL_PATH = "jaundice_mobilenetv3.onnx"

def test_model():
    print("Testing ONNX model loading...")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return False
    
    print(f"Model file exists: {os.path.getsize(MODEL_PATH)/1024/1024:.2f} MB")
    
    try:
        print("Creating inference session...")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 2
        
        session = ort.InferenceSession(MODEL_PATH, session_options, providers=['CPUExecutionProvider'])
        print("Model loaded successfully!")
        
        # Get input details
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Model input name: {input_name}, shape: {input_shape}")
        
        # Create dummy input for test inference
        print("Running test inference...")
        dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"Test inference successful! Output shape: {outputs[0].shape}")
        print(f"Raw output value: {outputs[0]}")
        
        # Calculate prediction
        logit = outputs[0][0][0]
        probability = 1 / (1 + np.exp(-logit))  # Sigmoid
        print(f"Prediction probability: {probability:.4f}")
        print(f"Predicted class: {'Jaundice' if probability > 0.5 else 'Normal'}")
        
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    print("\n" + "="*40)
    if success:
        print("✅ Model test PASSED")
    else:
        print("❌ Model test FAILED")
    print("="*40)
    sys.exit(0 if success else 1)
EOF

chmod +x ~/jaundice_monitor/test_model.py

# Test the model
echo "Testing model loading and inference..."
python ~/jaundice_monitor/test_model.py

echo ""
echo "==== Setup Complete ===="
echo "The system is now configured to use a Python virtual environment."
echo ""
echo "To activate the virtual environment manually:"
echo "  source ~/jaundice_monitor/venv/bin/activate"
echo ""
echo "To start the service:"
echo "  sudo systemctl start jaundice_monitor.service"
echo ""
echo "To check service status:"
echo "  sudo systemctl status jaundice_monitor.service"
echo ""
echo "To deactivate the virtual environment when done:"
echo "  deactivate"
