#!/usr/bin/env python3
"""
Neonatal Jaundice Detection for Raspberry Pi
Modified version with improved camera handling
"""

import os
import time
import cv2
import numpy as np
import onnxruntime as ort
import argparse
from datetime import datetime
import subprocess

# Configuration
MODEL_PATH = "jaundice_mobilenetv3.onnx"
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ["Normal", "Jaundice"]
LOG_FILE = "jaundice_detection_log.txt"
SAVE_INTERVAL = 60  # Save detection images every 60 seconds (if jaundice detected)
FSWEBCAM_CAPTURE = True  # Use fswebcam as fallback for problematic cameras

def log_message(message):
    """Log message with timestamp to console and log file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(log_entry)
    
    with open(LOG_FILE, "a") as f:
        f.write(log_entry + "\n")

def capture_frame_fswebcam(camera_device="/dev/video0", resolution="640x480", timeout=3):
    """Capture frame using fswebcam command (more reliable for some cameras)"""
    temp_file = "temp_capture.jpg"
    try:
        # Ensure fswebcam is installed
        if not os.path.exists("/usr/bin/fswebcam"):
            log_message("Installing fswebcam...")
            subprocess.run(["sudo", "apt", "install", "-y", "fswebcam"], check=True)
        
        # Capture image with fswebcam
        cmd = ["fswebcam", "-d", camera_device, "-r", resolution, "--no-banner", temp_file]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Read the captured image
        if os.path.exists(temp_file):
            frame = cv2.imread(temp_file)
            os.remove(temp_file)  # Clean up
            if frame is not None:
                return True, frame
    except Exception as e:
        log_message(f"fswebcam error: {e}")
    
    return False, None

def preprocess_frame(frame):
    """Preprocess frame for model inference"""
    # Resize and crop
    h, w = frame.shape[:2]
    if h > w:
        new_h, new_w = int(h * IMG_SIZE / w), IMG_SIZE
    else:
        new_h, new_w = IMG_SIZE, int(w * IMG_SIZE / h)
        
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Center crop
    h, w = resized.shape[:2]
    start_x = max(0, w // 2 - IMG_SIZE // 2)
    start_y = max(0, h // 2 - IMG_SIZE // 2)
    cropped = resized[start_y:start_y + IMG_SIZE, start_x:start_x + IMG_SIZE]
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    
    # Normalize
    img_float = img_rgb.astype(np.float32) / 255.0
    normalized = (img_float - MEAN) / STD
    
    # Change to NCHW format for ONNX
    transposed = normalized.transpose(2, 0, 1)
    return np.expand_dims(transposed, axis=0)

def load_onnx_model(model_path):
    """Load ONNX model"""
    try:
        # Create ONNX Runtime session with optimizations for Raspberry Pi
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.intra_op_num_threads = 2  # Limit threads for Pi
        
        # Create inference session
        session = ort.InferenceSession(model_path, session_options, providers=['CPUExecutionProvider'])
        log_message(f"Model loaded successfully from {model_path}")
        return session
    except Exception as e:
        log_message(f"Error loading model: {e}")
        return None

def run_inference(session, input_tensor):
    """Run inference using ONNX Runtime"""
    try:
        # Get input and output names
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Run inference
        outputs = session.run([output_name], {input_name: input_tensor.astype(np.float32)})
        
        # Process output (sigmoid for binary classification)
        logit = outputs[0][0][0]
        probability = 1 / (1 + np.exp(-logit))  # Sigmoid
        
        predicted_class_idx = 1 if probability > 0.5 else 0
        predicted_class = CLASS_NAMES[predicted_class_idx]
        
        return predicted_class, probability
    except Exception as e:
        log_message(f"Inference error: {e}")
        return None, None

def save_detection_image(frame, prediction, probability, output_dir="detections"):
    """Save frame with detection result as image"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Add text to image
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    text = f"{prediction} ({probability:.2%})"
    color = (0, 0, 255) if prediction == "Jaundice" else (0, 255, 0)  # BGR
    
    # Make a copy of the frame to avoid modifying the original
    annotated_frame = frame.copy()
    cv2.putText(annotated_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Save image
    filename = f"{timestamp}_{prediction}_{int(probability*100)}.jpg"
    filepath = os.path.join(output_dir, filename)
    cv2.imwrite(filepath, annotated_frame)
    log_message(f"Saved detection image to {filepath}")

def try_camera_indices(max_index=5):
    """Try multiple camera indices and return the first working one"""
    for idx in range(max_index):
        log_message(f"Trying camera index {idx}...")
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret and frame is not None and frame.size > 0:
                log_message(f"Successfully opened camera at index {idx}")
                return idx
    return None

def main(camera_index=0, display=False, alert_threshold=0.7, continuous_mode=True):
    """Main function for jaundice detection"""
    # Initialize
    log_message("Starting Neonatal Jaundice Detection")
    log_message(f"Initial camera index: {camera_index}, Display: {display}, Alert threshold: {alert_threshold}")
    
    # Load model
    session = load_onnx_model(MODEL_PATH)
    if session is None:
        log_message("Failed to load model. Exiting.")
        return
    
    # Ensure camera device exists
    camera_device = f"/dev/video{camera_index}"
    if not os.path.exists(camera_device):
        log_message(f"Camera device {camera_device} not found. Trying to detect cameras...")
        detected_index = try_camera_indices()
        if detected_index is not None:
            camera_index = detected_index
            camera_device = f"/dev/video{camera_index}"
            log_message(f"Using detected camera at index {camera_index}")
        else:
            log_message("No working cameras detected. Exiting.")
            return
    
    # Try different capture methods
    opencv_capture_working = False
    fswebcam_capture_working = False
    
    # Test OpenCV capture
    log_message(f"Testing OpenCV camera capture at index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            opencv_capture_working = True
            log_message("OpenCV camera capture is working")
        else:
            log_message("OpenCV camera opened but failed to read frame")
        cap.release()
    else:
        log_message("Failed to open camera with OpenCV")
    
    # Test fswebcam capture if OpenCV failed
    if not opencv_capture_working:
        log_message("Testing fswebcam capture as fallback...")
        ret, frame = capture_frame_fswebcam(camera_device)
        if ret and frame is not None and frame.size > 0:
            fswebcam_capture_working = True
            log_message("fswebcam capture is working, will use as fallback")
        else:
            log_message("fswebcam capture also failed")
    
    if not opencv_capture_working and not fswebcam_capture_working:
        log_message("All camera capture methods failed. Please check camera connection and permissions.")
        return
    
    # Camera setup for Raspberry Pi - lower resolution for better performance
    if opencv_capture_working:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    last_save_time = 0
    try:
        while True:
            # Capture frame
            if opencv_capture_working:
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    log_message("Failed to grab frame with OpenCV. Trying fswebcam fallback...")
                    if fswebcam_capture_working:
                        ret, frame = capture_frame_fswebcam(camera_device)
                    else:
                        log_message("No working capture method available. Retrying in 5 seconds...")
                        time.sleep(5)
                        continue
            else:
                # Use fswebcam directly
                ret, frame = capture_frame_fswebcam(camera_device)
            
            if not ret or frame is None or frame.size == 0:
                log_message("Failed to grab frame with all methods. Retrying in 5 seconds...")
                time.sleep(5)
                continue
            
            # Preprocess frame
            input_tensor = preprocess_frame(frame)
            
            # Run inference
            prediction, probability = run_inference(session, input_tensor)
            
            if prediction is None:
                continue
            
            # Display detection result
            text = f"{prediction} ({probability:.2%})"
            color = (0, 0, 255) if prediction == "Jaundice" else (0, 255, 0)  # BGR
            
            # Log high probability jaundice detections
            if prediction == "Jaundice" and probability >= alert_threshold:
                log_message(f"ALERT: High probability jaundice detected: {probability:.2%}")
                
                # Save images at intervals to avoid filling storage
                current_time = time.time()
                if current_time - last_save_time >= SAVE_INTERVAL:
                    save_detection_image(frame, prediction, probability)
                    last_save_time = current_time
            
            # Display results if requested
            if display:
                display_frame = frame.copy()
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.imshow("Jaundice Detection", display_frame)
                
                # Exit on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # Brief status update (not too frequent to avoid log spam)
                if int(time.time()) % 60 == 0:  # Log once per minute
                    log_message(f"Running... Last detection: {prediction} ({probability:.2%})")
            
            # If not in continuous mode, exit after one detection
            if not continuous_mode:
                break
                
    except KeyboardInterrupt:
        log_message("Detection stopped by user")
    except Exception as e:
        log_message(f"Error in main loop: {e}")
    finally:
        # Clean up
        if opencv_capture_working and cap.isOpened():
            cap.release()
        if display:
            cv2.destroyAllWindows()
        log_message("Jaundice detection stopped")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Neonatal Jaundice Detection for Raspberry Pi")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--display", action="store_true", help="Display video feed (not recommended for headless)")
    parser.add_argument("--threshold", type=float, default=0.7, help="Alert threshold for jaundice probability (default: 0.7)")
    parser.add_argument("--single", action="store_true", help="Single detection mode (exit after one detection)")
    parser.add_argument("--fswebcam", action="store_true", help="Force using fswebcam for capture")
    
    args = parser.parse_args()
    
    # Force fswebcam if requested
    if args.fswebcam:
        FSWEBCAM_CAPTURE = True
    
    main(
        camera_index=args.camera,
        display=args.display,
        alert_threshold=args.threshold,
        continuous_mode=not args.single
    )
