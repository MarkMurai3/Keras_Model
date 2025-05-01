import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import confusion_matrix
import seaborn as sns

from src.LeNetModel import LeNetModel
from src.SimpleCNNModel import SimpleCNNModel

from src.MiniVGGModel import MiniVGGModel
from src.DropoutMLPModel import DropoutMLPModel


# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load preprocessed EMNIST dataset
x_train = np.load("data/independent_test_set/x_train_emnist.npy")
y_train = np.load("data/independent_test_set/y_train_emnist.npy")
x_test = np.load("data/independent_test_set/x_test_emnist.npy")
y_test = np.load("data/independent_test_set/y_test_emnist.npy")

# Reshape and normalize
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

# Hyperparameter configurations
optimizers = [Adam, SGD, RMSprop]
learning_rates = [0.001, 0.0001]
epochs_list = [5, 10]

# Models to train
model_classes = [LeNetModel, SimpleCNNModel, MiniVGGModel, DropoutMLPModel]

results = []

for model_class in model_classes:
    for optimizer_class in optimizers:
        for lr in learning_rates:
            for epochs in epochs_list:
                model_name = model_class.__name__
                print(f"\nTraining {model_name} with optimizer={optimizer_class.__name__}, learning_rate={lr}, epochs={epochs}")

                # Initialize and build model
                model = model_class()
                model.build_model()
                model.compile_model(
                    optimizer=optimizer_class(learning_rate=lr),
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy']
                )

                # Train and save plot
                plot_path = f"plots/training_{model_name}_{optimizer_class.__name__}_lr{lr}_ep{epochs}.png"
                model.train_model(
                    x_train=x_train,
                    y_train=y_train,
                    batch_size=32,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    save_plot_path=plot_path
                )

                # Save model
                save_path = f"models/{model_name}_{optimizer_class.__name__}_lr{lr}_ep{epochs}.h5"
                model.save_model(save_path)
                print(f"Model saved to {save_path}")

                # Evaluate
                test_loss, test_acc = model.evaluate_model(x_test, y_test)
                print(f"Test Accuracy: {test_acc:.4f}, Loss: {test_loss:.4f}")

                results.append({
                    "Model": model_name,
                    "Optimizer": optimizer_class.__name__,
                    "Learning Rate": lr,
                    "Epochs": epochs,
                    "Test Accuracy": test_acc,
                    "Test Loss": test_loss
                })

                # Confusion matrix
                y_pred = model.model.predict(x_test).argmax(axis=1)
                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
                plt.title(f"Confusion Matrix - {model_name} - {optimizer_class.__name__} - lr={lr} - epochs={epochs}")
                plt.xlabel("Predicted")
                plt.ylabel("True")
                cm_path = f"plots/conf_matrix_{model_name}_{optimizer_class.__name__}_lr{lr}_ep{epochs}.png"
                plt.savefig(cm_path)
                plt.close()

# Save summary CSV
df = pd.DataFrame(results)
df.to_csv("plots/emnist_results_summary.csv", index=False)
print("All results saved to plots/emnist_results_summary.csv")
