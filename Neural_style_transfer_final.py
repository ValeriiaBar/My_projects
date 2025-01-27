import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# CONSTANTS
TARGET_SIZE = (224, 224)
OUTPUT_QUALITY = 50
STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
CONTENT_LAYERS = ['block5_conv2']


def load_and_compress_image(image_path: str, target_size: tuple, output_quality: int, save_path: str) -> Image.Image:
    """
    Load, resize, and compress an image.
    """
    img = Image.open(image_path)
    img_resized = img.resize(target_size)
    img_resized.save(save_path, format="JPEG", quality=output_quality, optimize=True)
    return img_resized


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess the image for VGG19.
    """
    image_array = np.expand_dims(image, axis=0)
    return keras.applications.vgg19.preprocess_input(image_array)


def deprocess_image(processed_image: np.ndarray) -> np.ndarray:
    """
    Deprocess the image back to a displayable format.
    """
    x = processed_image.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    return np.clip(x, 0, 255).astype('uint8')


def initialize_vgg19(style_layers: list, content_layers: list) -> keras.models.Model:
    """
    Initialize a VGG19 model pre-trained on ImageNet and configure outputs for style and content layers.
    """
    vgg = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in style_layers + content_layers]
    return keras.models.Model(inputs=vgg.input, outputs=outputs)


def get_feature_representations(model: keras.models.Model, style_image: np.ndarray, content_image: np.ndarray) -> tuple:
    """
    Extract style and content features from the model.
    """
    outputs = model(style_image)
    style_features = [style_layer[0] for style_layer in outputs[:len(STYLE_LAYERS)]]
    content_features = [content_layer[0] for content_layer in outputs[len(STYLE_LAYERS):]]
    return style_features, content_features


# Main pipeline
def main():
    content_image_path = r"C:\Users\Lenovo\Downloads\Telegram Desktop\me_img.jpg"
    style_image_path = r"C:\Users\Lenovo\Downloads\style_anime_img.jpg"

    compressed_content_path = "compressed_content.jpg"
    compressed_style_path = "compressed_style.jpg"

    # Step 1: Load and compress images
    content_image = load_and_compress_image(content_image_path, TARGET_SIZE, OUTPUT_QUALITY, compressed_content_path)
    style_image = load_and_compress_image(style_image_path, TARGET_SIZE, OUTPUT_QUALITY, compressed_style_path)

    print(f"Content image size: {os.path.getsize(compressed_content_path)} bytes")
    print(f"Style image size: {os.path.getsize(compressed_style_path)} bytes")

    # Step 2: Display images
    plt.subplot(1, 2, 1)
    plt.imshow(content_image)
    plt.title("Content Image")
    plt.subplot(1, 2, 2)
    plt.imshow(style_image)
    plt.title("Style Image")
    plt.show()

    # Step 3: Preprocess images
    x_content = preprocess_image(content_image)
    x_style = preprocess_image(style_image)
    print("Images preprocessed for VGG19.")

    # Step 4: Initialize model
    model = initialize_vgg19(STYLE_LAYERS, CONTENT_LAYERS)
    print("VGG19 initialized.")

    # Step 5: Extract features
    style_features, content_features = get_feature_representations(model, x_style, x_content)
    print("Style and content features extracted.")


if __name__ == "__main__":
    main()
