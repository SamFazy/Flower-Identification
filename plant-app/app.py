import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from google import genai
from google.genai import types
import json
import os
from dotenv import load_dotenv

# Get API key from .env file
load_dotenv()
APIKEY = os.getenv("GEMINI_API_KEY")

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get class names
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load Resnet50 model
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(class_names))
)

model.load_state_dict(torch.load("model.pth", map_location=device))
model = model.to(device)
model.eval()

# Transforms for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Prediction function
def predict(image):
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    return class_names[pred]

# Google GenAI Client
client = genai.Client(api_key=APIKEY)

def get_flower_info(flower_name):
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""
        You are a botanist.

        Flower: {flower_name}

        In Point form, provide the following information about the flower:
        - Care tips
        - Origin
        - Interesting facts
        - Simple description
        - If its perennial or annual
        - How much sunlight it needs
        - How much water it needs
        """
    )

    return response.text

# Streamlit App
st.title("🌿 Plant Identifier App 🌿")

uploaded_file = st.file_uploader("Upload a plant image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width=300)

    st.write("Classifying...")

    prediction = predict(image)

    st.success(f"Prediction: {prediction}")

    st.write("Getting plant info from AI...")

    info = get_flower_info(prediction)

    st.write(info)