import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def predict_image(model_path, image_path):
    # Load pre-trained model
    model = load_model(model_path)

    # Load and preprocess the image
    img = load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)
    predicted_label = np.argmax(prediction)
    print(f"Predicted Label: {predicted_label}")
