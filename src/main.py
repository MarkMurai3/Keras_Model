# Import necessary libraries
# import tensorflow as tf
# from matplotlib.pyplot import imshow
from tensorflow.keras.datasets import mnist
# from tensorflow.keras.utils import to_categorical
from src.LeNetModel import LeNetModel
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns



#.venv\Scripts\activate
#python -m src.main


#Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train.reshape(-1, 28, 28, 1) / 255.0, x_test.reshape(-1, 28, 28, 1) / 255.0

# Instantiate the LeNet model
# lenet = LeNetModel()

# Store results in a list
results = []


# # Build, compile, and summarize the model
# lenet.build_model()
# lenet.compile_model()
# lenet.summary()
#
# # Train the model
# lenet.train_model(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=32, save_plot_path="plots/training_plot.png")

optimizers = [Adam, SGD, RMSprop]
learning_rates = [0.01, 0.001, 0.0001]
epochs_list = [5, 10]

def visualize_mnist_samples(x_train, y_train, num_samples=10):
    import matplotlib.pyplot as plt

    # Plot MNIST samples
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_train[i].reshape(28, 28), cmap="gray")
        plt.title(f"Label: {y_train[i]}")
        plt.axis("off")
    plt.show()

#visualize_emnist_samples()

# Add this after loading MNIST data
visualize_mnist_samples(x_train, y_train)

from src.fine_tune import fine_tune_model

# Paths
pretrained_model_path = "models/lenet_adam_lr0.001_epochs10.h5"
emnist_data_path = "data/independent_test_set"
fine_tuned_model_path = "models/lenet_emnist_fine_tuned.h5"

# Fine-tune the model
fine_tune_model(
    pretrained_model_path=pretrained_model_path,
    emnist_data_path=emnist_data_path,
    fine_tuned_model_path=fine_tuned_model_path,
    epochs=5,
    batch_size=32
)


# Train models with different configurations
for optimizer_class in optimizers:
    for lr in learning_rates:
        for epochs in epochs_list:
            print(f"\nTraining with optimizer={optimizer_class.__name__}, learning_rate={lr}, epochs={epochs}")

            # Create a new instance of LeNetModel
            lenet = LeNetModel()

            # Build and compile the model
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

            # Test the model and evaluate
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



# Save the results table to a CSV file
results_df = pd.DataFrame(results)
results_csv_path = "plots/results_summary.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to {results_csv_path}")
