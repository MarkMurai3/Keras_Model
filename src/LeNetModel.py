from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from src.BaseModel import BaseModel

class LeNetModel(BaseModel):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.__input_shape = input_shape
        self.__num_classes = num_classes
        self.model = None
        self.memory = None  # Simulating dynamic memory allocation
        print(f"LeNetModel created with input shape: {self.__input_shape} and num classes: {self.__num_classes}")

    def __del__(self):
        print(f"LeNetModel with input shape: {self.__input_shape} and num classes: {self.__num_classes} is being destroyed")

    def allocate_memory(self, size):
        """Simulate dynamic memory allocation."""
        self.memory = [0] * size  # Allocate a list of the given size
        print(f"Allocated memory of size {size}")

    def free_memory(self):
        """Simulate freeing allocated memory."""
        self.memory = None
        print("Memory freed")

    @property
    def input_shape(self):
        return self.__input_shape

    @input_shape.setter
    def input_shape(self, value):
        if not isinstance(value, tuple):
            raise ValueError("Input shape must be a tuple")
        self.__input_shape = value

    @property
    def num_classes(self):
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, value):
        if not isinstance(value, int) or value <= 0:
            raise ValueError("Number of classes must be a positive integer")
        self.__num_classes = value

    @classmethod
    def from_dict(cls, config):
        """Conversion constructor: Create an instance from a dictionary."""
        input_shape = config.get("input_shape", (28, 28, 1))
        num_classes = config.get("num_classes", 10)
        return cls(input_shape, num_classes)

    def copy(self):
        """Copy constructor: Create a duplicate of the current instance."""
        return LeNetModel(self.__input_shape, self.__num_classes)

    @staticmethod
    def cast_to_lenet_model(obj):
        """Dynamic casting: Validate and cast an object to LeNetModel."""
        if isinstance(obj, LeNetModel):
            return obj
        else:
            raise TypeError("Cannot cast object to LeNetModel. Object is not of the correct type.")

    def build_model(self):
        input_layer = Input(shape=self.__input_shape)
        x = Conv2D(filters=6, kernel_size=(5, 5), activation='relu', padding='same')(input_layer)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Conv2D(filters=16, kernel_size=(5, 5), activation='relu')(x)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = Flatten()(x)
        x = Dense(units=120, activation='relu')(x)
        x = Dense(units=84, activation='relu')(x)
        output_layer = Dense(units=self.__num_classes, activation='softmax')(x)

        self.model = Model(inputs=input_layer, outputs=output_layer)

    def compile_model(self, optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy']):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    def train_model(self, x_train, y_train, validation_data, epochs, batch_size, save_plot_path):
        history = self.model.fit(
            x_train, y_train, validation_data=validation_data, epochs=epochs, batch_size=batch_size
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
