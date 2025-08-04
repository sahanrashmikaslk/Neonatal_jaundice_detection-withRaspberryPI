#!/bin/bash
# Troubleshooting script for Neonatal Jaundice Monitoring Service
# This script helps diagnose issues with the monitoring service

echo "==== Jaundice Monitor Troubleshooting ===="

# Check if files exist
echo "Checking for required files..."
if [ -f ~/jaundice_monitor/raspberry_monitor.py ]; then
    echo "✓ raspberry_monitor.py exists"
else
    echo "✗ raspberry_monitor.py is missing!"
fi

if [ -f ~/jaundice_monitor/jaundice_mobilenetv3.onnx ]; then
    echo "✓ jaundice_mobilenetv3.onnx exists"
else
    echo "✗ jaundice_mobilenetv3.onnx is missing!"
fi

# Check logs
echo -e "\nChecking service logs..."
echo "Last 10 lines of service.log:"
tail -n 10 ~/jaundice_monitor/service.log 2>/dev/null || echo "service.log not found or empty"

# Check Python dependencies
echo -e "\nChecking Python dependencies..."
echo "OpenCV version:"
python3 -c "import cv2; print(cv2.__version__)" 2>/dev/null || echo "OpenCV not installed or has errors"

echo "ONNX Runtime version:"
python3 -c "import onnxruntime as ort; print(ort.__version__)" 2>/dev/null || echo "ONNX Runtime not installed or has errors"

echo "NumPy version:"
python3 -c "import numpy as np; print(np.__version__)" 2>/dev/null || echo "NumPy not installed or has errors"

# Create a minimal test script to test model loading
echo -e "\nCreating minimal test script..."
cat > ~/jaundice_monitor/test_model.py << 'EOF'
import os
import sys
import onnxruntime as ort

# Check paths
model_path = os.path.join(os.path.expanduser("~"), "jaundice_monitor", "jaundice_mobilenetv3.onnx")
print(f"Looking for model at: {model_path}")
print(f"File exists: {os.path.isfile(model_path)}")

try:
    # Try to load the model
    print("Attempting to load model...")
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
    
    # Get input and output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Model loaded successfully!")
    print(f"Input name: {input_name}")
    print(f"Output name: {output_name}")
    print("Test successful!")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)
EOF

echo -e "\nRunning minimal test script..."
python3 ~/jaundice_monitor/test_model.py

# Try to run the actual script with debugging
echo -e "\nAttempting to run raspberry_monitor.py with verbose output..."
echo "This may show the actual error:"
python3 ~/jaundice_monitor/raspberry_monitor.py --camera 0 2>&1 | head -n 20

echo -e "\n==== Troubleshooting Complete ===="
echo "If the above tests show errors, consider the following solutions:"
echo "1. Check if the model file was properly transferred"
echo "2. Ensure all Python dependencies are installed"
echo "3. Make sure the camera is properly connected"
echo "4. Check permissions on the files and directories"
echo ""
echo "You can manually run the script with:"
echo "  python3 ~/jaundice_monitor/raspberry_monitor.py --camera 0"
echo ""
echo "To view the full service logs:"
echo "  cat ~/jaundice_monitor/service.log"
