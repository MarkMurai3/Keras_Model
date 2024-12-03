import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import History
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from src.BaseModel import BaseModel

class LeNetModel(BaseModel):
    def __init__(self, input_shape=(28, 28, 1), num_classes=10):
        self.__input_shape = input_shape
        self.__num_classes = num_classes
        self.model = None

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
