#!/bin/bash
# Troubleshoot virtual environment setup for Jaundice Monitor

echo "==== Virtual Environment Troubleshooting ===="

# Check if virtual environment exists
if [ -d ~/jaundice_monitor/venv ]; then
    echo "✓ Virtual environment exists at ~/jaundice_monitor/venv"
else
    echo "✗ Virtual environment not found at ~/jaundice_monitor/venv"
    echo "  Run simple_install.sh to create it"
fi

# Check for required files
if [ -f ~/jaundice_monitor/raspberry_monitor.py ]; then
    echo "✓ raspberry_monitor.py exists"
else
    echo "✗ raspberry_monitor.py not found"
fi

if [ -f ~/jaundice_monitor/jaundice_mobilenetv3.onnx ]; then
    echo "✓ jaundice_mobilenetv3.onnx exists"
    echo "  File size: $(du -h ~/jaundice_monitor/jaundice_mobilenetv3.onnx | cut -f1)"
else
    echo "✗ jaundice_mobilenetv3.onnx not found"
fi

# Test virtual environment
echo ""
echo "Testing virtual environment..."
if [ -f ~/jaundice_monitor/venv/bin/python ]; then
    # Activate virtual environment
    source ~/jaundice_monitor/venv/bin/activate
    
    echo "Python version:"
    python --version
    
    echo ""
    echo "Checking packages in virtual environment:"
    pip list | grep -E 'numpy|onnxruntime|opencv'
    
    echo ""
    echo "Testing imports:"
    python -c "
try:
    import numpy
    print('✓ NumPy import successful')
except ImportError as e:
    print('✗ NumPy import failed:', e)

try:
    import cv2
    print('✓ OpenCV import successful')
except ImportError as e:
    print('✗ OpenCV import failed:', e)

try:
    import onnxruntime
    print('✓ ONNX Runtime import successful')
except ImportError as e:
    print('✗ ONNX Runtime import failed:', e)
"

    # Try to run the model
    echo ""
    echo "Testing model loading (may take a moment)..."
    cd ~/jaundice_monitor
    python -c "
import os
import numpy as np
try:
    import onnxruntime as ort
    if os.path.exists('jaundice_mobilenetv3.onnx'):
        print('Model file exists')
        session = ort.InferenceSession('jaundice_mobilenetv3.onnx', providers=['CPUExecutionProvider'])
        print('✓ Model loaded successfully')
        dummy_input = np.zeros((1, 3, 224, 224), dtype=np.float32)
        outputs = session.run(None, {session.get_inputs()[0].name: dummy_input})
        print('✓ Inference test passed')
    else:
        print('✗ Model file not found')
except Exception as e:
    print('✗ Error:', e)
"

    # Deactivate virtual environment
    deactivate
else
    echo "✗ Virtual environment python not found"
fi

# Check service file
echo ""
echo "Checking service configuration..."
if [ -f /etc/systemd/system/jaundice_monitor.service ]; then
    echo "Service file exists. Contents:"
    echo "------------------------"
    cat /etc/systemd/system/jaundice_monitor.service
    echo "------------------------"
    
    # Check if service is using virtual environment
    if grep -q "venv/bin/python" /etc/systemd/system/jaundice_monitor.service; then
        echo "✓ Service is configured to use virtual environment"
    else
        echo "✗ Service is NOT configured to use virtual environment"
    fi
else
    echo "✗ Service file not found in /etc/systemd/system/"
fi

echo ""
echo "==== Troubleshooting Complete ===="
echo ""
echo "If you see any errors above, try running simple_install.sh again"
echo "or manually install the missing packages in the virtual environment."
echo ""
echo "To manually start the monitor (for testing):"
echo "  source ~/jaundice_monitor/venv/bin/activate"
echo "  cd ~/jaundice_monitor"
echo "  python raspberry_monitor.py --camera 0"
