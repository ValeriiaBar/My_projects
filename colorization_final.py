import os
from tensorflow.keras.layers import Conv2D, UpSampling2D, InputLayer
from tensorflow.keras import Sequential
from skimage.color import rgb2lab, lab2rgb
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set TensorFlow environment variable to disable OneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Constants
IMAGE_PATH = r"C:\Users\Lenovo\Downloads\cats400.jpg"
IMAGE_SIZE = (256, 256)
EPOCHS = 50
BATCH_SIZE = 1


def load_image(image_path: str) -> Image.Image:
    """
    Load an image from the specified path.
    :param image_path: Path to the image file
    :return: Loaded PIL Image
    """
    return Image.open(image_path)


def preprocess_image(img: Image.Image, target_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """
    Resize and preprocess the image for training.
    Converts RGB to LAB color space and splits into L (grayscale) and AB (color) channels.
    :param img: Input PIL Image
    :param target_size: Target size for resizing the image
    :return: Tuple containing processed L channel, AB channels, and the original size
    """
    image = img.resize(target_size, Image.Resampling.BILINEAR)
    image = np.array(image, dtype=float)
    original_size = image.shape

    # Convert RGB to LAB color space
    lab_image = rgb2lab(image / 255.0)
    X = lab_image[:, :, 0]  # L channel
    Y = lab_image[:, :, 1:]  # AB channels

    # Normalize AB channels
    Y /= 128

    # Reshape for model input
    X = X.reshape(1, target_size[0], target_size[1], 1)
    Y = Y.reshape(1, target_size[0], target_size[1], 2)

    return X, Y, original_size


def build_model() -> Sequential:
    """
    Build and compile the colorization model.
    :return: Compiled Keras Sequential model
    """
    model = Sequential([
        InputLayer(shape=(None, None, 1)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        Conv2D(64, (3, 3), activation='relu', padding='same', strides=2),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same', strides=2),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same', strides=2),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        UpSampling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        Conv2D(2, (3, 3), activation='tanh', padding='same'),
        UpSampling2D((2, 2))
    ])

    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model: Sequential, X: np.ndarray, Y: np.ndarray, epochs: int, batch_size: int) -> None:
    """
    Train the model on the given data.
    :param model: Keras model to train
    :param X: Input L channel data
    :param Y: Target AB channel data
    :param epochs: Number of training epochs
    :param batch_size: Batch size for training
    """
    model.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs)


def predict_and_visualize(model: Sequential, X: np.ndarray, original_size: tuple[int, int], img: Image.Image) -> None:
    """
    Predict the AB channels using the model and visualize the result.
    :param model: Trained Keras model
    :param X: Input L channel data
    :param original_size: Original image size
    :param img: Original PIL Image
    """
    # Predict AB channels
    output = model.predict(X)
    output *= 128  # Denormalize AB channels

    # Clip values and prepare LAB image
    ab = np.clip(output[0], -128, 127)
    lab_image = np.zeros((original_size[0], original_size[1], 3))
    lab_image[:, :, 0] = np.clip(X[0][:, :, 0], 0, 100)  # L channel
    lab_image[:, :, 1:] = ab

    # Convert LAB to RGB
    rgb_image = lab2rgb(lab_image)

    # Visualize results
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_image)
    plt.title("Processed Image")

    plt.show()


def main():
    # Load and preprocess the image
    img = load_image(IMAGE_PATH)
    X, Y, original_size = preprocess_image(img, IMAGE_SIZE)

    # Build and train the model
    model = build_model()
    train_model(model, X, Y, EPOCHS, BATCH_SIZE)

    # Predict and visualize the result
    predict_and_visualize(model, X, original_size, img)


if __name__ == '__main__':
    main()
