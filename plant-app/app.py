import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from google import genai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API key from .env
env_api_key = os.getenv("GEMINI_API_KEY")

# Session state for manual API key and invalid key flag
if "manual_api_key" not in st.session_state:
    st.session_state.manual_api_key = ""

if "api_key_invalid" not in st.session_state:
    st.session_state.api_key_invalid = False

# Determine which API key to use (manual takes precedence over .env)
APIKEY = st.session_state.manual_api_key or env_api_key

# If no key or invalid key, prompt user to enter one
if not env_api_key or st.session_state.api_key_invalid:

    if st.session_state.api_key_invalid:
        st.error("The Gemini API key in the .env file is invalid.")
    else:
        st.warning("No Gemini API key found in .env")

    manual_key = st.text_input(
        "Enter a Gemini API Key for gemini-2.5-flash-lite",
        type="password"
    )

    if manual_key:

        # Save manual key
        st.session_state.manual_api_key = manual_key

        # Clear invalid flag
        st.session_state.api_key_invalid = False

        # Reload app with new key
        st.rerun()

# Gemini Client Initialization
client = None

if APIKEY:
    try:
        client = genai.Client(api_key=APIKEY)
    except:
        client = None


# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Class Names
with open("../class_names.json", "r") as f:
    class_names = json.load(f)

# Load Model
model = models.resnet50(weights=None)

model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, len(class_names))
)

model.load_state_dict(
    torch.load("../model.pth", map_location=device)
)

model = model.to(device)
model.eval()

# Image Transformers
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Predict function
def predict(image):

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        pred = torch.argmax(output, dim=1).item()

    return class_names[pred]


# Gemini API call to get flower info
def get_flower_info(flower_name):

    global client

    # No key available
    if client is None:
        return "No valid Gemini API key provided."

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=f"""
            You are a botanist.

            Flower: {flower_name}

            In point form provide:
            - Care tips
            - Origin
            - Interesting facts
            - Simple description
            - If perennial or annual
            - Sunlight needs
            - Water needs
            """
        )

        # API key worked
        st.session_state.api_key_invalid = False

        return response.text

    except Exception as e:

        error_text = str(e).lower()

        # Detect invalid API key
        if (
            "api key" in error_text
            or "authentication" in error_text
            or "permission" in error_text
            or "invalid" in error_text
        ):

            # Mark key as invalid
            st.session_state.api_key_invalid = True

            # Remove bad manual key if one exists
            st.session_state.manual_api_key = ""

            # Rerun app immediately
            st.rerun()

        return "Failed to get AI flower information."


# Streamlit App
st.title("🌿 Plant Identifier App 🌿")

uploaded_file = st.file_uploader(
    "Upload a plant image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    st.write("Classifying...")

    prediction = predict(image)

    st.success(f"Prediction: {prediction}")

    st.write("Getting plant info from AI...")

    info = get_flower_info(prediction)

    st.write(info)