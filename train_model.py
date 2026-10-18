
import os
import tensorflow as tf
from tensorflow.keras import layers, models

# Keep the output model beside this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "cifar10_cnn_model.keras")

print("Loading the MNIST dataset...")

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

# Use a small subset for faster training
x_train = x_train[:5000].astype("float32") / 255.0
y_train = y_train[:5000]

x_test = x_test[:1000].astype("float32") / 255.0
y_test = y_test[:1000]

# Add a channel dimension: 28 x 28 grayscale images
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print("Training images:", x_train.shape)

# CNN with 3 convolutional layers
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Training the model...")

model.fit(
    x_train,
    y_train,
    epochs=3,
    batch_size=64,
    validation_split=0.1
)

loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print(f"Test accuracy: {accuracy * 100:.2f}%")

model.save(MODEL_PATH)

print("Model saved successfully!")
print("Saved at:", MODEL_PATH)

