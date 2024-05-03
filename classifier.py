import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define constants
IMAGE_SIZE = 256
CHANNELS = 3
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']  # Updated class names

# Load the model
@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('potatoes.h5')
    return model

model = load_model()

# Function to preprocess image
def preprocess_image(image):
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = tf.cast(image, tf.float32)
    image /= 255.0  # Normalize pixel values
    return image

# Function to make predictions
def predict(model, img):
    img_array = preprocess_image(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Streamlit app
def main():
    st.title("Potato Disease Classifier")
    st.write("Upload an image of a potato leaf for disease classification")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded image and convert it to NumPy array
        image = tf.image.decode_image(uploaded_file.read(), channels=CHANNELS)
        image_np = np.array(image)

        # Display the uploaded image using Streamlit
        st.image(image_np, caption='Uploaded Image', use_column_width=True)

        # Make prediction on the uploaded image
        if st.button('Predict'):
            predicted_class, confidence = predict(model, image)
            st.write(f"Predicted class: {predicted_class}")
            st.write(f"Confidence: {confidence}%")

if __name__ == "__main__":
    main()
