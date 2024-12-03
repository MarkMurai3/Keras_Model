import os
import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model


class DrawingApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("Draw a Number")
        self.model = load_model(model_path)

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        # Buttons
        self.predict_button = tk.Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.clear_button = tk.Button(root, text="Clear", command=self.clear_canvas)
        self.clear_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Image to capture drawing
        self.image = Image.new("L", (28, 28), "black")  # 28x28 image, black background
        self.draw = ImageDraw.Draw(self.image)

        # Bind mouse events to canvas
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        """Draw on the canvas and update the internal image."""
        x, y = event.x, event.y
        self.canvas.create_oval(x, y, x + 10, y + 10, fill="black", width=0)
        scaled_x, scaled_y = x // 10, y // 10
        self.draw.ellipse([scaled_x, scaled_y, scaled_x + 1, scaled_y + 1], fill="white")

    def clear_canvas(self):
        """Clear the canvas and reset the internal image."""
        self.canvas.delete("all")
        self.image = Image.new("L", (28, 28), "black")
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        """Predict the digit based on the drawing."""
        img_array = np.array(self.image) / 255.0  # Normalize pixel values
        img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for the model

        prediction = self.model.predict(img_array)
        predicted_label = np.argmax(prediction)

        # Display the prediction
        result_window = tk.Toplevel(self.root)
        result_window.title("Prediction")
        result_label = tk.Label(result_window, text=f"Predicted Digit: {predicted_label}", font=("Arial", 24))
        result_label.pack(padx=20, pady=20)


if __name__ == "__main__":
    # Read the best model path from the file
    best_model_file = os.path.join("models", "best_model.txt")
    if not os.path.exists(best_model_file):
        raise FileNotFoundError("Best model file not found. Run model_evaluator.py first.")

    with open(best_model_file, "r") as f:
        model_path = f.read().strip()

    # Create the Tkinter window
    root = tk.Tk()
    app = DrawingApp(root, model_path)
    root.mainloop()
