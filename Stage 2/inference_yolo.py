import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load YOLOv5 pretrained model from the web (pretrained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Using YOLOv5 small model

# Load the pre-trained Sentence Transformer model
text_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Predefined menstrual waste categories for comparison (for generating target embeddings)
menstrual_categories = ["pad", "tampon", "sanitary panties", "menstrual cup", "packaging"]

def get_textual_embedding(labels):
    """
    Returns textual embeddings for the detected labels using Sentence Transformers.
    """
    embeddings = text_model.encode(labels)
    return embeddings

def compute_embedding_confidence(detected_embeddings, target_embeddings):
    """
    Compute cosine similarity between detected object embeddings and predefined target embeddings.
    This provides a confidence score for how well the detected objects match the intended categories.
    """
    similarity_scores = cosine_similarity(detected_embeddings, target_embeddings)
    return similarity_scores

def plot_image_with_detections(image_path, detected_objects, confidences, output_image_path):
    """
    Plots and saves the image with bounding boxes, object names, and confidence scores.
    """
    img = cv2.imread(image_path)

    for i, obj in enumerate(detected_objects):
        bbox = obj['bbox']
        label = obj['name']
        confidence = confidences[i]

        # Draw the bounding box
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)

        # Add the label and confidence score
        text = f"{label} {confidence:.2f}"
        cv2.putText(img, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Save the output image
    cv2.imwrite(output_image_path, img)

    # Plot original and result image side by side
    original_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    result_img = cv2.cvtColor(cv2.imread(output_image_path), cv2.COLOR_BGR2RGB)

    # Plot the images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(result_img)
    axes[1].set_title("Image with Detections")
    axes[1].axis("off")
    plt.show()

def detect_and_segregate(image_path, output_image_path='output_with_detections.jpg'):
    """
    Detects objects in the given image using pretrained YOLOv5 model and returns 
    the detected components along with confidence scores for textual embeddings.
    Additionally, saves the image with bounding boxes, labels, and confidence scores.
    """
    # Perform inference with YOLOv5
    results = model(image_path)
    
    # Extract detected labels, bounding boxes, and confidence scores
    detections = results.pandas().xyxy[0]  # Get results as pandas DataFrame
    detected_labels = detections['name'].tolist()
    confidence_scores = detections['confidence'].tolist()

    if not detected_labels:
        print("No objects detected.")
        return [], [], []

    # Generate textual embeddings for the detected labels
    detected_embeddings = get_textual_embedding(detected_labels)
    
    # Generate textual embeddings for target menstrual waste categories
    target_embeddings = get_textual_embedding(menstrual_categories)
    
    # Compute cosine similarity between detected and target embeddings
    embedding_confidences = compute_embedding_confidence(detected_embeddings, target_embeddings)

    # Prepare objects with bbox, name, and confidence score for image visualization
    detected_objects = [
        {
            'name': detections['name'][i],
            'bbox': [detections['xmin'][i], detections['ymin'][i], detections['xmax'][i], detections['ymax'][i]],
            'confidence': confidence_scores[i]
        } for i in range(len(detections))
    ]
    
    # Plot and save the result
    plot_image_with_detections(image_path, detected_objects, confidence_scores, output_image_path)

    return detected_labels, confidence_scores, embedding_confidences

# Example usage
image_path = r'C:\Users\Ritabrata\Tilottoma\dataset\menstrual\menstrual_waste_image_19.jpg'  # Replace with your test image path
output_image_path = r'C:\Users\Ritabrata\Tilottoma\output_with_detections.jpg'
labels, obj_confidences, text_confidences = detect_and_segregate(image_path, output_image_path)

# Output the detected labels, confidence scores, and embedding similarities
print("Detected Objects:", labels)
print("Object Detection Confidence Scores:", obj_confidences)
print("Text Embedding Confidence Scores (Cosine Similarity):", text_confidences)
