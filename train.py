import numpy as np
from data import load_mnist_dataset
from network import NeuralNetwork

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

# Initialize neural network with the new hyperparameters
nn = NeuralNetwork(learning_rate=0.0001, clip_value=5)

print("\nTraining on MNIST dataset...")
nn.train(X_train, y_train, epochs=10, batch_size=64)

np.savez("model_weights.npz",
         weights_input_hidden=nn.weights_input_hidden,
         bias_hidden=nn.bias_hidden,
         weights_hidden_output=nn.weights_hidden_output,
         bias_output=nn.bias_output)

print("\nTraining complete! Model saved as 'model_weights.npz'.")
