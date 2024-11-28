import os
import numpy as np
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

def fine_tune_model(pretrained_model_path, emnist_data_path, fine_tuned_model_path, epochs=15, batch_size=32):
    """
    Fine-tune a pre-trained model on the EMNIST dataset.

    Parameters:
        pretrained_model_path (str): Path to the pre-trained MNIST model.
        emnist_data_path (str): Path to the directory containing EMNIST .npy files.
        fine_tuned_model_path (str): Path to save the fine-tuned model.
        epochs (int): Number of fine-tuning epochs.
        batch_size (int): Batch size for training.

    Returns:
        None
    """
    # Load the EMNIST dataset
    x_train_emnist = np.load(os.path.join(emnist_data_path, "x_train_emnist.npy"))
    y_train_emnist = np.load(os.path.join(emnist_data_path, "y_train_emnist.npy"))
    x_test_emnist = np.load(os.path.join(emnist_data_path, "x_test_emnist.npy"))
    y_test_emnist = np.load(os.path.join(emnist_data_path, "y_test_emnist.npy"))

    print("EMNIST dataset loaded successfully.")

    # Load the pre-trained model
    model = load_model(pretrained_model_path)
    print("Pre-trained MNIST model loaded successfully.")

    # Freeze the first few layers
    for layer in model.layers[:-4]:  # Freeze all but the last 4 layers
        layer.trainable = False
    print("All but the last 4 layers are frozen for fine-tuning.")

    # Modify the output layer for EMNIST (10 classes)
    x = model.layers[-2].output  # Get the output of the second-to-last layer
    output_layer = Dense(units=10, activation='softmax', name="new_output")(x)
    model = Model(inputs=model.input, outputs=output_layer)
    print("Output layer modified for EMNIST.")

    # Compile the model with a small learning rate
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled for fine-tuning.")

    # Fine-tune the model
    history = model.fit(
        x_train_emnist, y_train_emnist,
        validation_data=(x_test_emnist, y_test_emnist),
        epochs=epochs,
        batch_size=batch_size
    )
    print("Fine-tuning completed.")

    # Save the fine-tuned model
    os.makedirs(os.path.dirname(fine_tuned_model_path), exist_ok=True)
    model.save(fine_tuned_model_path, include_optimizer=True)
    print(f"Fine-tuned model saved to {fine_tuned_model_path}.")

    # Evaluate the fine-tuned model
    loss, accuracy = model.evaluate(x_test_emnist, y_test_emnist, verbose=0)
    print(f"EMNIST Test Loss: {loss:.4f}, EMNIST Test Accuracy: {accuracy:.4f}")

    # Plot the training history
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Fine-Tuning Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    print("Fine-tuning process completed.")
