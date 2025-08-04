#!/bin/bash
# Fix dependencies for Raspberry Pi Jaundice Monitor
# This script installs all required dependencies properly

echo "==== Installing Dependencies for Jaundice Monitor ===="
echo "This may take some time on a Raspberry Pi 3B+..."

# Create log directory
mkdir -p ~/jaundice_monitor/logs

# First, update package lists
echo "[1/7] Updating package lists..."
sudo apt update

# Install system dependencies for OpenCV and ONNX Runtime
echo "[2/7] Installing system dependencies..."
sudo apt install -y python3-pip python3-dev cmake build-essential libatlas-base-dev \
    libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev \
    libswscale-dev libv4l-dev libxvidcore-dev libx264-dev \
    libjasper-dev libhdf5-dev libhdf5-serial-dev libhdf5-103 \
    libqt5gui5 libqt5webkit5 libqt5test5 python3-pyqt5 > ~/jaundice_monitor/logs/dependencies.log 2>&1

# Install Python dependencies - try directly through apt first for better compatibility
echo "[3/7] Installing Python packages from apt (faster and more compatible)..."
sudo apt install -y python3-opencv python3-numpy > ~/jaundice_monitor/logs/python_apt_install.log 2>&1

# Check if OpenCV was installed successfully, if not use pip
if ! python3 -c "import cv2" &> /dev/null; then
    echo "[4/7] OpenCV not installed via apt, trying pip (this may take 30+ minutes)..."
    pip3 install --upgrade pip
    pip3 install opencv-python-headless > ~/jaundice_monitor/logs/opencv_pip_install.log 2>&1
else
    echo "[4/7] OpenCV installed successfully via apt."
fi

# Check if NumPy was installed
if ! python3 -c "import numpy" &> /dev/null; then
    echo "NumPy not installed, installing via pip..."
    pip3 install numpy > ~/jaundice_monitor/logs/numpy_pip_install.log 2>&1
else
    echo "NumPy installed successfully."
fi

# Install ONNX Runtime (optimized for ARM)
echo "[5/7] Installing ONNX Runtime (this may take 10+ minutes)..."
pip3 install onnxruntime > ~/jaundice_monitor/logs/onnx_install.log 2>&1

# Verify installations
echo "[6/7] Verifying installations..."
echo "OpenCV: "
python3 -c "import cv2; print('OpenCV version:', cv2.__version__)" || echo "OpenCV installation failed"

echo "NumPy: "
python3 -c "import numpy; print('NumPy version:', numpy.__version__)" || echo "NumPy installation failed"

echo "ONNX Runtime: "
python3 -c "import onnxruntime; print('ONNX Runtime version:', onnxruntime.__version__)" || echo "ONNX Runtime installation failed"

# Create a test script to check if model loading works
echo "[7/7] Testing model loading..."
cat > ~/jaundice_monitor/test_model.py << 'EOF'
#!/usr/bin/env python3
import os
import sys
import numpy as np
import onnxruntime as ort

# Configuration
MODEL_PATH = "jaundice_mobilenetv3.onnx"

def test_model():
    print(f"Testing model loading from {MODEL_PATH}")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return False
    
    try:
        # Create ONNX Runtime session
        print("Creating inference session...")
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 2
        
        # Create session
        session = ort.InferenceSession(MODEL_PATH, session_options, providers=['CPUExecutionProvider'])
        
        # Check input shape
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print(f"Model input name: {input_name}, shape: {input_shape}")
        
        # Create dummy input
        dummy_input = np.random.rand(1, 3, 224, 224).astype(np.float32)
        
        # Run test inference
        print("Running test inference...")
        outputs = session.run(None, {input_name: dummy_input})
        
        print(f"Test inference successful! Output shape: {outputs[0].shape}")
        return True
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    if success:
        print("\n✅ Model loading test PASSED")
        sys.exit(0)
    else:
        print("\n❌ Model loading test FAILED")
        sys.exit(1)
EOF

# Make it executable
chmod +x ~/jaundice_monitor/test_model.py

# Run the test
cd ~/jaundice_monitor
python3 test_model.py

# Final message
echo ""
echo "==== Installation Complete ===="
if python3 -c "import cv2, numpy, onnxruntime" &> /dev/null; then
    echo "✅ All dependencies installed successfully!"
    echo ""
    echo "You can now start the monitor service with:"
    echo "  sudo systemctl start jaundice_monitor.service"
    echo ""
    echo "To check service status:"
    echo "  sudo systemctl status jaundice_monitor.service"
else
    echo "❌ Some dependencies could not be installed."
    echo "Check the logs in ~/jaundice_monitor/logs/ for details."
fi
