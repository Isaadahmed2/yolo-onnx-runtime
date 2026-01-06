import numpy as np
import cv2

def preprocess_frame(frame, input_width, input_height):
    """Preprocesses a single frame for YOLOv8 inference."""
    # Resize and pad the frame to the model's expected input size
    h, w, _ = frame.shape
    scale = min(input_width / w, input_height / h)
    scaled_w, scaled_h = int(w * scale), int(h * scale)
    scaled_frame = cv2.resize(frame, (scaled_w, scaled_h))

    # Create a black canvas of the target size and paste the scaled frame onto it
    padded_frame = np.full((input_height, input_width, 3), 0, dtype=np.uint8)
    top_pad = (input_height - scaled_h) // 2
    left_pad = (input_width - scaled_w) // 2
    padded_frame[top_pad:top_pad + scaled_h, left_pad:left_pad + scaled_w] = scaled_frame

    # Convert to float32, normalize, and transpose from HWC to CHW format
    image_data = np.array(padded_frame, dtype=np.float32) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))  # HWC to CHW
    image_data = np.expand_dims(image_data, axis=0)  # Add batch dimension

    return image_data, scale, left_pad, top_pad

def postprocess_output(output, scale, left_pad, top_pad, conf_threshold=0.5):
    """Postprocesses the YOLOv8 output to get bounding boxes."""
    # The output of YOLOv8 is [batch, 84, n] where 84 is class_probs + 4 box coords
    # We need to transpose it to [batch, n, 84]
    output = np.transpose(output[0], (1, 0))

    boxes = []
    for row in output:
        # The first 4 values are box coordinates (cx, cy, w, h)
        # The rest are class probabilities
        class_probs = row[4:]
        class_id = np.argmax(class_probs)
        confidence = class_probs[class_id]

        # COCO class for 'sports ball' is 32
        if confidence > conf_threshold and class_id == 32:
            cx, cy, w, h = row[:4]

            # Convert from center coordinates to x1, y1, x2, y2
            x1 = int((cx - w / 2 - left_pad) / scale)
            y1 = int((cy - h / 2 - top_pad) / scale)
            x2 = int((cx + w / 2 - left_pad) / scale)
            y2 = int((cy + h / 2 - top_pad) / scale)
            boxes.append((x1, y1, x2, y2))

    return boxes
