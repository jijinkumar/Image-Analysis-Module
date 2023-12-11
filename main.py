import cv2
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
import torch

# Function to detect dress color using Blip model
def detect_dress_color(image_path):
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Convert image to text using the Blip model
    inputs = processor(image, return_tensors="pt")
    outputs = model.generate(**inputs)
    dress_color = processor.decode(outputs[0], skip_special_tokens=True)

    return dress_color

# Function to detect faces and gender using Face API
def detect_faces_and_gender(image_path):
    # Use the Face Detection pipeline from Hugging Face Transformers
    face_pipeline = pipeline("face-detection", model="deepset/roberta-base-openai-detector")

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect faces using the pipeline
    faces = face_pipeline(image_rgb)

    # Process each detected face
    for face in faces:
        # Use a pre-trained model for gender classification
        gender_model = pipeline("image-classification", model="mrm8488/wider_gender_roberta")
        gender_prediction = gender_model(face["face"])
        gender = gender_prediction[0]["label"]

        print("Gender:", gender)

# Function to recognize objects (anime characters or animals) using a pre-trained object recognition model
def recognize_objects(image_path):
    # Load the object recognition model (replace with an appropriate model)
    object_recognition_model = torch.hub.load('facebookresearch/detectron2:main', 'faster_rcnn.resnet50')
    object_recognition_model.eval()

    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run object recognition
    predictions = object_recognition_model(image_rgb)

    # Extract and print recognized classes
    recognized_classes = predictions["instances"].pred_classes.tolist()
    print("Recognized Classes:", recognized_classes)

# Example usage
image_path = "pxfuel.jpg"

# Detect dress color
dress_color = detect_dress_color(image_path)
print('Dress Color:', dress_color)

# Detect faces and gender
detect_faces_and_gender(image_path)

# Recognize objects (anime characters or animals)
recognize_objects(image_path)
