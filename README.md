
# CNN CIFAR-10 Image Classification

## About the Project
This project uses a Convolutional Neural Network (CNN) to
classify images into 10 CIFAR-10 categories.

The project includes a trained deep learning model and an
interactive Streamlit frontend for image prediction.

## Technologies Used
- Python
- TensorFlow and Keras
- Streamlit
- NumPy
- Pillow
- CIFAR-10 dataset

## Classes
Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse,
Ship, and Truck.

## Features
- Upload JPG and PNG images
- Predict the image category
- Display prediction confidence
- View scores for all 10 classes
- Interactive web interface

## Project Structure
- app.py: Streamlit frontend
- cifar10_cnn_model.keras: Trained CNN model
- Welcome_To_Colab(10).ipynb: Model training notebook
- requirements.txt: Python dependencies

## Run Locally

1. Install Python.
2. Install dependencies:

   pip install -r requirements.txt

3. Ensure the trained model is in the project folder.
4. Start the application:

   streamlit run app.py

5. Open the local URL shown in your terminal.

## Note
The CIFAR-10 model accepts 32x32 RGB images.
Predictions on arbitrary real-world photographs may be
less accurate because the model was trained on CIFAR-10.
