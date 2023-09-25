import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

# Set page title and description
st.set_page_config(
    page_title="Fire vs. No Fire Classifier",
    page_icon="🔥",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Page title and header
st.title("[Demo with Transfer Learning] Fire vs. No Fire Image Classifier")

st.write(
    "This app uses a pre-trained EfficientNet-B0 model for image classification."
)

# Upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)

    # Load the pre-trained model from TensorFlow Hub
    model_url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/feature_vector/2"
    model = tf.keras.Sequential([
        hub.KerasLayer(model_url, trainable=False),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

    # Load and preprocess the image
    img = load_img(uploaded_image, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Make predictions with the model
    predictions = model.predict(img_array)
    class_names = ['Non Fire', 'Fire']
    label = class_names[np.argmax(predictions)]
    confidence = predictions[0][np.argmax(predictions)]

    # Display prediction result in a colored box
    st.subheader("Prediction Result:")
    prediction_color = "red" if label == "Fire" else "green"
    st.markdown(
        f'<div style="background-color: {prediction_color}; padding: 10px; border-radius: 5px; color: white;">'
        f'<h3>{label} (Confidence: {confidence:.2f})</h3></div>',
        unsafe_allow_html=True,
    )

# Adding the moving text with green color
st.markdown(
    '<marquee behavior="scroll" direction="left" style="color: green; font-size: 16px;">Developed by Said Al Afghani Edsa</marquee>',
    unsafe_allow_html=True,
)

st.success("Please note that this model is a pre-trained EfficientNet-B0 model fine-tuned for 'Fire' vs. 'No Fire' classification.")
