import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10):
        """
        Initializes a simple feedforward neural network.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Weight Initialization: Small random values
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, self.hidden_size))

        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.bias_output = np.zeros((1, self.output_size))

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

# Test the network architecture
if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.summary()
