import numpy as np
from data import load_mnist_dataset
from network import NeuralNetwork
import matplotlib.pyplot as plt

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = load_mnist_dataset()

# Convert labels to one-hot encoding
def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

y_train = one_hot_encode(y_train)
y_test = one_hot_encode(y_test)

# Initialize neural network with your tuned hyperparameters
nn = NeuralNetwork(learning_rate=0.0001, clip_value=5)

# Train on MNIST data
print("\nTraining on MNIST dataset...")
nn.train(X_train, y_train, epochs=10, batch_size=64)

# Save trained model parameters (optional)
np.savez("model_weights.npz",
         weights_input_hidden=nn.weights_input_hidden,
         bias_hidden=nn.bias_hidden,
         weights_hidden_output=nn.weights_hidden_output,
         bias_output=nn.bias_output)
print("\nTraining complete! Model saved as 'model_weights.npz'.")

# --- Evaluation ---

def evaluate_model(nn, X, y):
    """
    Evaluates the neural network on test data and prints the accuracy.
    Expects y to be one-hot encoded.
    """
    y_pred = nn.forward_propagation(X)
    predicted_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y, axis=1)
    
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return predicted_labels, true_labels

predicted_labels, true_labels = evaluate_model(nn, X_test, y_test)

def visualize_predictions(X, true_labels, predicted_labels, num_samples=10):
    """
    Visualizes a number of test images with their true and predicted labels.
    """
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 2.5))
    for i in range(num_samples):
        img = X[i].reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"True: {true_labels[i]}\nPred: {predicted_labels[i]}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# Visualize predictions on the first 10 test images
visualize_predictions(X_test, true_labels, predicted_labels, num_samples=10)
