import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: rgb(26, 28, 36);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    .upload-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .result-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.keras')

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make prediction
def predict_tumor(image):
    model = load_model()
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return prediction[0][0]

# Header
st.title("🧠 Brain Tumor Detection")
st.markdown("""
    This application uses deep learning to detect brain tumors in MRI images.
    Upload an MRI image to get started.
    """)

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    
    # Make prediction
    with st.spinner('Analyzing the image...'):
        prediction = predict_tumor(image)
        
        # Display results
        st.markdown("### Results")
        if prediction > 0.5:
            st.error(f"⚠️ Tumor Detected (Confidence: {prediction*100:.2f}%)")
            st.markdown("""
                **Please consult with a medical professional for further evaluation.**
                This is an AI-based prediction and should not be used as a definitive diagnosis.
                """)
        else:
            st.success(f"✅ No Tumor Detected (Confidence: {(1-prediction)*100:.2f}%)")
            st.markdown("""
                **Note:** This is an AI-based prediction and should not be used as a definitive diagnosis.
                Always consult with a medical professional for proper evaluation.
                """)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>This application is for educational purposes only.</p>
        <p>Always consult with medical professionals for proper diagnosis.</p>
    </div>
    """, unsafe_allow_html=True) 