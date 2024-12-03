import os
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Paths to the EMNIST dataset files
images_path = "emnist/emnist-digits-test-images-idx3-ubyte"
labels_path = "emnist/emnist-digits-test-labels-idx1-ubyte"

# Output directory for processed files
output_dir = "data/independent_test_set/"
os.makedirs(output_dir, exist_ok=True)

def load_emnist_images(images_path):
    """Load and preprocess EMNIST images."""
    with open(images_path, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # Magic number
        num_images = int.from_bytes(f.read(4), 'big')  # Number of images
        rows = int.from_bytes(f.read(4), 'big')  # Rows
        cols = int.from_bytes(f.read(4), 'big')  # Columns
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, rows, cols, 1) / 255.0  # Normalize
    return images

def load_emnist_labels(labels_path):
    """Load EMNIST labels."""
    with open(labels_path, 'rb') as f:
        _ = int.from_bytes(f.read(4), 'big')  # Magic number
        num_labels = int.from_bytes(f.read(4), 'big')  # Number of labels
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

def transform_images(images):
    """Mirror and rotate images to the left by 90 degrees."""
    transformed_images = []
    for image in images:
        # Reshape image for transformation
        image = image.reshape(28, 28)
        # Mirror the image
        mirrored_image = np.fliplr(image)
        # Rotate the image 90 degrees to the left
        rotated_image = np.rot90(mirrored_image)
        # Append the transformed image
        transformed_images.append(rotated_image.reshape(28, 28, 1))
    return np.array(transformed_images)

# Load the EMNIST dataset
x_emnist = load_emnist_images(images_path)
y_emnist = load_emnist_labels(labels_path)

# Transform the images
x_emnist = transform_images(x_emnist)

# Debugging: Check EMNIST normalization and label range
print(f"EMNIST max pixel value: {x_emnist.max()}, min pixel value: {x_emnist.min()}")
print(f"EMNIST unique labels: {np.unique(y_emnist)}")

# Visualize some transformed EMNIST samples
# for i in range(5):
#     plt.imshow(x_emnist[i].reshape(28, 28), cmap='gray')
#     plt.title(f"Label: {y_emnist[i]}")
#     plt.axis('off')
#     plt.show()

# Split into training and test sets
x_train_emnist, x_test_emnist, y_train_emnist, y_test_emnist = train_test_split(
    x_emnist, y_emnist, test_size=0.2, random_state=42
)

# Save the processed dataset as .npy files
np.save(os.path.join(output_dir, "x_train_emnist.npy"), x_train_emnist)
np.save(os.path.join(output_dir, "y_train_emnist.npy"), y_train_emnist)
np.save(os.path.join(output_dir, "x_test_emnist.npy"), x_test_emnist)
np.save(os.path.join(output_dir, "y_test_emnist.npy"), y_test_emnist)

print(f"EMNIST dataset split, transformed, and saved to {output_dir}")
