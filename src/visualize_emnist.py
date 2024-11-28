import numpy as np
import matplotlib.pyplot as plt
import os

# Paths to the processed EMNIST data
x_new_path = "data/independent_test_set/x_new.npy"
y_new_path = "data/independent_test_set/y_new.npy"

def visualize_emnist_samples(x_path, y_path, num_samples=10):
    # Load EMNIST data
    x_new = np.load(x_path)
    y_new = np.load(y_path)

    # Plot a few samples
    plt.figure(figsize=(10, 2))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(x_new[i].reshape(28, 28), cmap="gray")
        plt.title(f"Label: {y_new[i]}")
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    visualize_emnist_samples(x_new_path, y_new_path)
