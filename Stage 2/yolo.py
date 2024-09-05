import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

def detect_and_segregate(image_path):
    """
    Detects objects in the given image using pretrained YOLOv5 model and returns 
    the detected components along with confidence scores for textual embeddings.
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

    # Display results (bounding boxes and labels on the image)
    results.show()

    return detected_labels, confidence_scores, embedding_confidences

# Example usage
image_path = 'test_image.jpg'  # Replace with your test image path
labels, obj_confidences, text_confidences = detect_and_segregate(image_path)

# Output the detected labels, confidence scores, and embedding similarities
print("Detected Objects:", labels)
print("Object Detection Confidence Scores:", obj_confidences)
print("Text Embedding Confidence Scores (Cosine Similarity):", text_confidences)
