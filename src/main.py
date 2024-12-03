import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from sklearn.metrics import confusion_matrix
import seaborn as sns
from src.LeNetModel import LeNetModel

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28, 28, 1) / 255.0, x_test.reshape(-1, 28, 28, 1) / 255.0

# Hyperparameter configurations
optimizers = [Adam, SGD, RMSprop]
learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [5, 10]

# Results list for storing evaluation metrics
results = []

# Train models with different configurations
for optimizer_class in optimizers:
    for lr in learning_rates:
        for epochs in epochs_list:
            print(f"\nTraining with optimizer={optimizer_class.__name__}, learning_rate={lr}, epochs={epochs}")

            # Create and configure the model
            lenet = LeNetModel()
            lenet.build_model()
            lenet.compile_model(
                optimizer=optimizer_class(learning_rate=lr),
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )

            # Generate a unique plot file name
            plot_path = f"plots/training_plot_{optimizer_class.__name__}_lr{lr}_epochs{epochs}.png"

            # Train the model
            lenet.train_model(
                x_train=x_train,
                y_train=y_train,
                batch_size=32,
                epochs=epochs,
                validation_data=(x_test, y_test),
                save_plot_path=plot_path
            )

            # Save the trained model
            model_name = f"lenet_{optimizer_class.__name__}_lr{lr}_epochs{epochs}.h5"
            model_save_path = f"models/{model_name}"
            lenet.model.save(model_save_path)
            print(f"Model saved to {model_save_path}")

            # Test the model
            test_loss, test_accuracy = lenet.model.evaluate(x_test, y_test, verbose=0)
            print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

            # Append results to the table
            results.append({
                'Optimizer': optimizer_class.__name__,
                'Learning Rate': lr,
                'Epochs': epochs,
                'Test Accuracy': test_accuracy,
                'Test Loss': test_loss
            })

            # Generate confusion matrix
            y_pred = lenet.model.predict(x_test).argmax(axis=1)
            cm = confusion_matrix(y_test, y_pred)

            # Plot confusion matrix
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
            plt.title(f"Confusion Matrix: {optimizer_class.__name__}, LR={lr}, Epochs={epochs}")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")

            # Save the confusion matrix plot
            cm_plot_path = f"plots/confusion_matrix_{optimizer_class.__name__}_lr{lr}_epochs{epochs}.png"
            plt.savefig(cm_plot_path)
            plt.close()
            print(f"Confusion matrix saved to {cm_plot_path}")

# Save results as a CSV file
results_df = pd.DataFrame(results)
results_csv_path = "plots/results_summary.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")
