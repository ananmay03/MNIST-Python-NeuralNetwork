import numpy as np
import struct
import os
import matplotlib.pyplot as plt

# Define dataset path
DATASET_PATH = "data/"

def load_mnist_images(filename):
    """"Load MNIST image data from a binary file."""
    filepath = os.path.join(DATASET_PATH, filename)
    print(f"Looking for file at: {os.path.abspath(filepath)}")  # Debugging output

    with open(filepath, 'rb') as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols)  # Flatten 28x28 images
        return images.astype(np.float32) / 255.0  # Normalize pixels to [0,1]

def load_mnist_labels(filename):
    """Load MNIST label data from a binary file."""
    with open(os.path.join(DATASET_PATH, filename), 'rb') as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def load_mnist_dataset():
    """Load both training and test MNIST datasets using the correct file names."""
    X_train = load_mnist_images("train-images.idx3-ubyte")
    y_train = load_mnist_labels("train-labels.idx1-ubyte")
    X_test = load_mnist_images("t10k-images.idx3-ubyte")
    y_test = load_mnist_labels("t10k-labels.idx1-ubyte") 
    return (X_train, y_train), (X_test, y_test)

def show_sample_images(X, y, num_samples=5):
    """Displays sample MNIST images with labels."""
    fig, axes = plt.subplots(1, num_samples, figsize=(10, 3))
    for i in range(num_samples):
        img = X[i].reshape(28, 28)  # Reshape back to 28x28
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Label: {y[i]}")
        axes[i].axis("off")
    plt.show()

if __name__ == "__main__":
    # Load dataset
    (X_train, y_train), (X_test, y_test) = load_mnist_dataset()
    
    # Print dataset info
    print(f"Training Data: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test Data: {X_test.shape}, Labels: {y_test.shape}")
    
    # Show some sample images
    show_sample_images(X_train, y_train)
