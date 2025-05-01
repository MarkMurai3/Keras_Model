from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.BaseModel import BaseModel

class MiniVGGModel(BaseModel):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.__input_shape = input_shape
        self.__num_classes = num_classes
        self.model = None
        print(f"MiniVGGModel created with input shape: {self.__input_shape} and num classes: {self.__num_classes}")

    def build_model(self):
        input_layer = Input(shape=self.__input_shape)

        # Block 1
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        # Block 2
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2))(x)

        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output_layer = Dense(self.__num_classes, activation='softmax')(x)

        self.model = Model(inputs=input_layer, outputs=output_layer)

    def compile_model(self, optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, x_train, y_train, validation_data, epochs, batch_size, save_plot_path):
        history = self.model.fit(
            x_train, y_train, validation_data=validation_data, epochs=epochs, batch_size=batch_size
        )

        # Plot training history
        import matplotlib.pyplot as plt
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(save_plot_path)
        plt.close()

    def evaluate_model(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, verbose=0)

    def save_model(self, filepath):
        self.model.save(filepath)
