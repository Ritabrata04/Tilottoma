"""
This stage is after Yolo. We take the results from Yolo and try to filter some of the objects detected
with low confidence.According to our closed feedback loop, we can manually segregate that then.
"""

import torch

# Load YOLOv5 pretrained model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Pretrained model

# Function to run YOLO detection
def run_yolo(image_path):
    results = model(image_path)  # Run YOLO on the image
    detections = results.pandas().xyxy[0]  # Get results in pandas DataFrame

    # Extracting labels, confidence scores, and bounding boxes
    detected_items = detections[['name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax']].to_dict(orient="records")
    
    return detected_items
