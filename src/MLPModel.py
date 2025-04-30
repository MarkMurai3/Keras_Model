# src/MLPModel.py

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from src.BaseModel import BaseModel

class MLPModel(BaseModel):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None

    def build_model(self):
        inputs = Input(shape=self.input_shape)
        x = Flatten()(inputs)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        outputs = Dense(self.num_classes, activation='softmax')(x)

        self.model = Model(inputs=inputs, outputs=outputs)

    def compile_model(self, optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, x_train, y_train, validation_data, epochs, batch_size, save_plot_path):
        history = self.model.fit(
            x_train, y_train, validation_data=validation_data,
            epochs=epochs, batch_size=batch_size
        )

        # Plot training results
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
