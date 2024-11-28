from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt


class LeNetModel:
    def __init__(self, input_shape=(28, 28, 1), num_classes=10): #28 = height, 28 = width, 1 = The number of color channels (in this case, 1 for grayscale images).
        # Initialize the class with input shape and number of classes
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None  # Placeholder for the model

    def build_model(self):
        # Define the input layer
        input_layer = Input(shape=self.input_shape)

        # First convolutional layer
        x = Conv2D(filters=6, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

        # Second convolutional layer
        x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

        # Flatten the layers to connect to fully connected (dense) layers
        x = Flatten()(x)

        # First fully connected layer
        x = Dense(units=120, activation='relu')(x)

        # Second fully connected layer
        x = Dense(units=84, activation='relu')(x)

        # Output layer with softmax activation
        output_layer = Dense(units=self.num_classes, activation='softmax')(x)

        # Define and assign the model
        self.model = Model(inputs=input_layer, outputs=output_layer)

    def compile_model(self, optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        # Ensure the model is built before compiling
        if not self.model:
            self.build_model()
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, x_train, y_train, batch_size=32, epochs=10, validation_data=None, save_plot_path="plots/training_plot.png"):
        # Ensure the model is compiled before training
        if not self.model:
            raise ValueError("Model must be compiled before training.")

        history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data
        )
        import os

        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        # Plot training metrics
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title("Training Metrics")
        plt.xlabel("Epochs")
        plt.ylabel("Metrics")
        plt.legend()
        plt.savefig(save_plot_path)
        plt.close()  # Close the plot to avoid overlapping in future runs

        print(f"Training plot saved to {save_plot_path}")
        return history

    def summary(self):
        # Display the model architecture
        if self.model:
            self.model.summary()
        else:
            print("The model has not been built yet.")

