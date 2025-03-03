import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load Trained Model
model = tf.keras.models.load_model("signature_verification_model.h5")  

# Streamlit UI
st.title("Signature Verification Model App")
st.write("Upload a signature image to check if it's genuine or forged.")

# Upload Image
uploaded_file = st.file_uploader("Choose a signature image...", type=["jpg", "png", "jpeg"])

# Preprocessing Function
def preprocess_image(image):
    image = image.resize((217, 217))  
    image = image.convert("L")  # Converts into grayscale
    image = np.array(image) / 255.0 
    image = np.expand_dims(image, axis=-1) 
    image = np.expand_dims(image, axis=0)  
    return image

# Prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB") 
    st.image(image, caption="Uploaded Signature", use_column_width=True)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    if st.button("Predict"):
        prediction = model.predict(processed_image)[0][0]  # Get probability
        st.write(f"Prediction Probability: {prediction:.4f}")

        
        # Convert to binary class (i have chosen threshold = 0.45)
        label = "Genuine Signature " if prediction >= 0.25 else "Forged Signature"
        st.success(f"Predicted Class: {label}")

