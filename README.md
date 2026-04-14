Flower Identifier App

A deep learning web app that classifies flower images and provides useful plant information.

Built using **PyTorch**, **ResNet50**, and **Streamlit**. This project allows users to upload an image of a flower and receive:
- Predicted flower name
- AI-generated plant information (care, origin, facts, etc.)

---

##Features

- Image classification using a fine-tuned **ResNet50**
- Trained on the **Oxford Flowers 102 dataset**
- Filters dataset to ensure class balance (≥ 50 images per class)
- Achieves ~**81% test accuracy**
- Interactive UI with **Streamlit**
- AI-generated plant descriptions using **Google Gemini API**

---

##Model Details

- Base Model: `ResNet50 (pretrained)`
- Fine-tuned layers: `layer3`, `layer4`, and `fc`
- Image size: `224x224`
- Optimizer: `Adam`
- Loss: `CrossEntropyLoss`
- Early stopping used to prevent overfitting

---

##Final Performance

- **Test Accuracy:** 81.43%
- **Precision:** 0.81
- **Recall:** 0.81
- **F1 Score:** 0.81

---

##How to Run the App

1. Clone the repository

```bash
git clone https://github.com/your-username/flower-identifier.git
cd flower-identifier
```

2. Install dependencies

```pip install torch torchvision streamlit datasets pillow python-dotenv google-generativeai```

3. Set up environment variables
  - Create .env file that contains: GEMINI_API_KEY=your_api_key_here

4. Run App
   - streamlit run app.py


Author: Samuel Faseruk
