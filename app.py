
import os
import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="CNN Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "mnist_cnn_model.keras"
)

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

st.title("🔢 CNN Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit from 0 to 9.")

uploaded_file = st.file_uploader(
    "Choose a digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("L")

    st.subheader("Your uploaded image")
    st.image(image, width=250)

    if st.button("Predict Digit", type="primary"):
        try:
            model = get_model()

            # MNIST images have a black background and
            # light digit strokes. Invert typical white
            # paper / dark ink images for MNIST-style input.
            image = ImageOps.autocontrast(image)
            image = image.resize((28, 28))

            array = np.array(image, dtype=np.float32)

            # Convert dark ink on a light background
            # to light strokes on a dark background.
            if array.mean() > 127:
                array = 255.0 - array

            array = array / 255.0
            array = array.reshape(1, 28, 28, 1)

            probabilities = model.predict(array, verbose=0)[0]
            digit = int(np.argmax(probabilities))

            st.success(f"Predicted digit: {digit}")
            st.metric(
                "Model confidence",
                f"{probabilities[digit] * 100:.2f}%"
            )

            st.subheader("Scores for digits 0–9")
            for i, probability in enumerate(probabilities):
                st.write(f"Digit {i}: {probability:.2%}")
                st.progress(float(probability))

            st.caption(
                "This demo works best with a single, centered "
                "handwritten digit on a plain background."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")

elif not os.path.exists(MODEL_PATH):
    st.warning(
        "The trained model is missing. Run "
        "'py train_model.py' first."
    )

