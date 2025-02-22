import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.01):
        """
        Initializes a simple feedforward neural network.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Weight Initialization: Small random values
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))

    def relu(self, x):
        """ReLU Activation Function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def softmax(self, x):
        """Softmax Activation Function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # Stability trick
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, y_true, y_pred):
        """Computes the cross-entropy loss"""
        m = y_true.shape[0]  # Number of examples
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m  # Adding small value for numerical stability
        return loss

    def forward_propagation(self, X):
        """
        Performs forward propagation and returns output probabilities.
        """
        # Compute hidden layer activation
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)

        # Compute output layer activation
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.softmax(self.output_layer_input)

        return self.output_layer_output

    def backpropagation(self, X, y_true, y_pred):
        """
        Performs backpropagation and updates weights and biases.
        """
        m = X.shape[0]  # Batch size

        # Compute error for output layer
        delta_output = (y_pred - y_true) / m  # Shape: (batch_size, 10)

        # Compute gradient for weights from hidden to output
        grad_weights_hidden_output = np.dot(self.hidden_layer_output.T, delta_output)
        grad_bias_output = np.sum(delta_output, axis=0, keepdims=True)

        # Compute error for hidden layer
        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_layer_input)

        # Compute gradient for weights from input to hidden
        grad_weights_input_hidden = np.dot(X.T, delta_hidden)
        grad_bias_hidden = np.sum(delta_hidden, axis=0, keepdims=True)

        # Update weights and biases using Gradient Descent
        self.weights_hidden_output -= self.learning_rate * grad_weights_hidden_output
        self.bias_output -= self.learning_rate * grad_bias_output
        self.weights_input_hidden -= self.learning_rate * grad_weights_input_hidden
        self.bias_hidden -= self.learning_rate * grad_bias_hidden

    def train(self, X, y, epochs=10):
        """
        Trains the neural network using the dataset.
        """
        for epoch in range(epochs):
            y_pred = self.forward_propagation(X)  # Forward pass
            loss = self.cross_entropy_loss(y, y_pred)  # Compute loss
            self.backpropagation(X, y, y_pred)  # Backpropagation

            if epoch % 1 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

    def summary(self):
        """
        Prints a summary of the network structure and parameters.
        """
        print("Neural Network Architecture")
        print("---------------------------")
        print(f"Input Layer: {self.input_size} neurons")
        print(f"Hidden Layer: {self.hidden_size} neurons (ReLU activation)")
        print(f"Output Layer: {self.output_size} neurons (Softmax activation)")
        print(f"Total Parameters: {self.total_parameters()}")

    def total_parameters(self):
        """
        Computes the total number of trainable parameters in the network.
        """
        total = (self.input_size * self.hidden_size) + self.hidden_size  # Input to Hidden Layer
        total += (self.hidden_size * self.output_size) + self.output_size  # Hidden to Output Layer
        return total

# Test the network with a dummy training run
if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.summary()

    # Create a dummy dataset (10 samples, 784 features)
    X_train = np.random.rand(10, 784)
    y_train = np.zeros((10, 10))
    y_train[np.arange(10), np.random.randint(0, 10, size=10)] = 1  # Random one-hot labels

    nn.train(X_train, y_train, epochs=5)
