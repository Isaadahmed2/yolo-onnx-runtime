import cv2
import onnxruntime
import numpy as np
from .utils import preprocess_frame, postprocess_output
import os
from .logging_config import get_logger

logger = get_logger(__name__)


MODEL_PATH = "yolo_model/yolov8n.onnx"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load the ONNX model
session = onnxruntime.InferenceSession(MODEL_PATH)

# Get model input details
model_inputs = session.get_inputs()
input_shape = model_inputs[0].shape
input_width = input_shape[3]
input_height = input_shape[2]


def draw_glowing_ball(frame, box):
    """Draws a 'glowing' effect around the detected ball."""
    x1, y1, x2, y2 = box

    # Extract the region of interest (the ball)
    ball_roi = frame[y1:y2, x1:x2]
    if ball_roi.size == 0:
        return

    # Create a blurred version of the ball for the glow effect
    # The size of the kernel for blurring should be odd
    kernel_size = max(1, (x2 - x1) // 4)
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred_roi = cv2.GaussianBlur(ball_roi, (kernel_size, kernel_size), 0)

    # Blend the blurred glow back onto the original frame
    # This creates a soft glow around the ball's edges
    glow_region = frame[y1:y2, x1:x2]
    cv2.addWeighted(blurred_roi, 1.5, glow_region, 0.5, 0, glow_region)

    # Draw a circle in the center of the bounding box for a highlight
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    radius = (x2 - x1) // 4
    cv2.circle(frame, (center_x, center_y), radius, (200, 255, 200), -1)


def process_video_with_yolo(input_path: str, output_path: str):
    """
    Processes a video to detect sports balls and apply a glowing effect.
    """
    logger.info(f"Starting video processing for {input_path}")
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        logger.error(f"Could not open video file {input_path}")
        return None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor, scale, left_pad, top_pad = preprocess_frame(frame, input_width, input_height)
        outputs = session.run(None, {model_inputs[0].name: input_tensor})
        boxes = postprocess_output(outputs[0], scale, left_pad, top_pad)

        # Apply glowing effect for each detected ball
        for box in boxes:
            draw_glowing_ball(frame, box)

        out.write(frame)

    cap.release()
    out.release()
    logger.info(f"Finished video processing. Output saved to {output_path}")
    return output_path

