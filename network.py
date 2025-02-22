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
        """ReLU Activation Function: max(0, x)"""
        return np.maximum(0, x)

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

# Test the network with a dummy input and dummy label
if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.summary()

    # Create a dummy input (1 sample, 784 pixels)
    dummy_input = np.random.rand(1, 784)
    output_probs = nn.forward_propagation(dummy_input)

    # Dummy one-hot encoded label (e.g., correct label is '3')
    dummy_label = np.zeros((1, 10))
    dummy_label[0, 3] = 1  # Setting the correct class to 1

    # Compute loss
    loss = nn.cross_entropy_loss(dummy_label, output_probs)

    print("\nSample Prediction (Probabilities):")
    print(output_probs)
    print("\nPredicted Digit:", np.argmax(output_probs))
    print("\nCross-Entropy Loss:", loss)
