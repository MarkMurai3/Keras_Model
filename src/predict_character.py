import argparse
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


def preprocess_image(image_path):
    """
    Preprocess the input image to match the model's input format.

    Parameters:
        image_path (str): Path to the input image.

    Returns:
        np.ndarray: Preprocessed image ready for prediction.
    """
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image not found at {image_path}")

    # Resize the image to 28x28 pixels
    img_resized = cv2.resize(img, (28, 28))

    # Normalize pixel values to [0, 1]
    img_normalized = img_resized / 255.0

    # Reshape the image to (28, 28, 1) for the model
    img_reshaped = img_normalized.reshape(1, 28, 28, 1)

    return img_reshaped


def predict_character(image_path, model_path):
    """
    Predict the character in the input image using the trained model.

    Parameters:
        image_path (str): Path to the input image.
        model_path (str): Path to the trained model.

    Returns:
        int: Predicted character (0-9).
    """
    # Preprocess the input image
    preprocessed_image = preprocess_image(image_path)

    # Load the trained model
    model = load_model(model_path)

    # Perform prediction
    prediction = model.predict(preprocessed_image)
    predicted_label = np.argmax(prediction)

    return predicted_label


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict the character in an input image using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("model_path", type=str, help="Path to the trained model.")
    args = parser.parse_args()

    # Perform prediction
    try:
        predicted_character = predict_character(args.image_path, args.model_path)
        print(f"Predicted Character: {predicted_character}")
    except Exception as e:
        print(f"Error: {e}")
