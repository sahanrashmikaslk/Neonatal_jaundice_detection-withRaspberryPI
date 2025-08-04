#!/usr/bin/env python3
"""
Simplified Neonatal Jaundice Detection for Raspberry Pi
Reduced dependencies and improved error handling
"""

import os
import time
import sys
import cv2
import numpy as np
import onnxruntime as ort
from datetime import datetime

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.expanduser("~"), "jaundice_monitor", "jaundice_mobilenetv3.onnx")
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ["Normal", "Jaundice"]
LOG_FILE = os.path.join(os.path.expanduser("~"), "jaundice_monitor", "jaundice_detection_log.txt")

# --- Logging ---
def log_message(message):
    """Log message with timestamp to console and log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    try:
        with open(LOG_FILE, "a") as f:
            f.write(log_entry + "\n")
    except Exception as e:
        print(f"Warning: Could not write to log file: {e}")

# --- Preprocessing ---
def preprocess_frame(frame):
    """Simplified preprocessing for model inference"""
    try:
        # Resize
        h, w = frame.shape[:2]
        scale = IMG_SIZE / min(h, w)
        new_size = (int(w * scale), int(h * scale))
        resized = cv2.resize(frame, new_size)
        
        # Center crop
        y_center, x_center = resized.shape[0] // 2, resized.shape[1] // 2
        y_start = max(0, y_center - IMG_SIZE // 2)
        x_start = max(0, x_center - IMG_SIZE // 2)
        cropped = resized[y_start:y_start + IMG_SIZE, x_start:x_start + IMG_SIZE]
        
        # Handle potential size mismatch
        if cropped.shape[0] != IMG_SIZE or cropped.shape[1] != IMG_SIZE:
            log_message(f"Warning: Crop size mismatch. Padding/resizing to {IMG_SIZE}x{IMG_SIZE}")
            result = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            h, w = min(cropped.shape[0], IMG_SIZE), min(cropped.shape[1], IMG_SIZE)
            result[:h, :w] = cropped[:h, :w]
            cropped = result
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        
        # Normalize
        img_float = img_rgb.astype(np.float32) / 255.0
        for i in range(3):
            img_float[:,:,i] = (img_float[:,:,i] - MEAN[i]) / STD[i]
        
        # Change to NCHW format for ONNX
        transposed = img_float.transpose(2, 0, 1)
        return np.expand_dims(transposed, axis=0)
    
    except Exception as e:
        log_message(f"Error in preprocessing: {e}")
        return None

# --- Model Loading ---
def load_onnx_model():
    """Load ONNX model with explicit error handling"""
    try:
        log_message(f"Attempting to load model from {MODEL_PATH}")
        
        if not os.path.exists(MODEL_PATH):
            log_message(f"Error: Model file not found at {MODEL_PATH}")
            return None
        
        # Create ONNX Runtime session with optimization
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 2  # Limit threads for Pi
        
        # Create inference session
        log_message("Creating inference session...")
        session = ort.InferenceSession(
            MODEL_PATH, 
            session_options, 
            providers=['CPUExecutionProvider']
        )
        
        # Verify input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        log_message(f"Model loaded successfully. Input: {input_name}, Output: {output_name}")
        
        return session
    except Exception as e:
        log_message(f"Error loading model: {str(e)}")
        return None

# --- Inference ---
def run_inference(session, input_tensor):
    """Run inference with error handling"""
    if session is None or input_tensor is None:
        return None, None
    
    try:
        # Get input name
        input_name = session.get_inputs()[0].name
        
        # Run inference
        outputs = session.run(None, {input_name: input_tensor.astype(np.float32)})
        
        # Process output (sigmoid for binary classification)
        logit = outputs[0][0][0]
        probability = 1 / (1 + np.exp(-logit))  # Sigmoid
        
        predicted_class_idx = 1 if probability > 0.5 else 0
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        return predicted_class, probability
    except Exception as e:
        log_message(f"Inference error: {str(e)}")
        return None, None

# --- Main Function ---
def main():
    """Main function with simplified camera handling"""
    log_message("Starting Simplified Neonatal Jaundice Detection")
    
    # Check for command line args
    camera_index = 0
    if len(sys.argv) > 2 and sys.argv[1] == "--camera":
        try:
            camera_index = int(sys.argv[2])
        except ValueError:
            log_message(f"Invalid camera index: {sys.argv[2]}. Using default: 0")
    
    # Load model
    session = load_onnx_model()
    if session is None:
        log_message("Failed to load model. Exiting.")
        sys.exit(1)
    
    # Open camera
    log_message(f"Opening camera at index {camera_index}")
    try:
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            log_message(f"Failed to open camera at index {camera_index}. Exiting.")
            sys.exit(1)
        
        # Camera setup - lower resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Main loop
        while True:
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                log_message("Failed to grab frame. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # Preprocess frame
            input_tensor = preprocess_frame(frame)
            if input_tensor is None:
                log_message("Preprocessing failed. Skipping frame.")
                continue
            
            # Run inference
            prediction, probability = run_inference(session, input_tensor)
            
            if prediction is None:
                log_message("Prediction failed. Skipping frame.")
                continue
            
            # Log detection result
            log_message(f"Detection: {prediction} ({probability:.2%})")
            
            # Sleep to reduce CPU usage
            time.sleep(2)
            
    except KeyboardInterrupt:
        log_message("Detection stopped by user")
    except Exception as e:
        log_message(f"Error in main loop: {str(e)}")
        sys.exit(1)
    finally:
        # Clean up
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        log_message("Jaundice detection stopped")

if __name__ == "__main__":
    main()
