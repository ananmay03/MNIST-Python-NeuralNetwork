import numpy as np
import matplotlib.pyplot as plt
from data import load_mnist_dataset
from network import NeuralNetwork

def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

def evaluate_model(nn, X, y):
    """
    Evaluates the neural network on test data and returns predicted and true labels.
    Expects y to be one-hot encoded.
    """
    y_pred = nn.forward_propagation(X)
    predicted_labels = np.argmax(y_pred, axis=1)
    true_labels = np.argmax(y, axis=1)
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    return predicted_labels, true_labels

def visualize_predictions_grid(X, true_labels, predicted_labels, grid_size=(10, 10), filename="example_visualization.png"):
    """
    Visualizes images in a grid with their true and predicted labels.
    
    Parameters:
      X: Array of images (each image should be 784-dimensional, reshaped to 28x28)
      true_labels: Array of true labels (integers)
      predicted_labels: Array of predicted labels (integers)
      grid_size: Tuple (rows, columns) for the grid display
    """
    n_rows, n_cols = grid_size
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < X.shape[0]:
                img = X[idx].reshape(28, 28)
                axes[i, j].imshow(img, cmap='gray')
                # Increase the padding so the text isn't cut off
                axes[i, j].set_title(f"T:{true_labels[idx]}\nP:{predicted_labels[idx]}", fontsize=8, pad=5)
                axes[i, j].axis('off')
    plt.tight_layout(pad=2.0)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = load_mnist_dataset()
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    # Initialize neural network with your chosen hyperparameters
    nn = NeuralNetwork(learning_rate=0.00025, clip_value=10, hidden_size=512)
    
    # Train on MNIST data
    print("\nTraining on MNIST dataset...")
    nn.train(X_train, y_train, epochs=20, batch_size=32)
    
    # Save trained model parameters (optional)
    np.savez("model_weights.npz",
             weights_input_hidden=nn.weights_input_hidden,
             bias_hidden=nn.bias_hidden,
             weights_hidden_output=nn.weights_hidden_output,
             bias_output=nn.bias_output)
    print("\nTraining complete! Model saved as 'model_weights.npz'.")
    
    # Evaluate the model on the test set
    predicted_labels, true_labels = evaluate_model(nn, X_test, y_test)
    
    # Visualize a grid of 100 test images with their true and predicted labels
    visualize_predictions_grid(X_test, true_labels, predicted_labels, grid_size=(10, 10))
