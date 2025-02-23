import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.0001, clip_value=5):
        """
        Initializes a simple feedforward neural network.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.clip_value = clip_value

        # He Initialization for ReLU
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.bias_output = np.zeros((1, self.output_size))
    
    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU."""
        return (x > 0).astype(float)
    
    def softmax(self, x):
        """Softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def cross_entropy_loss(self, y_true, y_pred):
        """Computes the cross-entropy loss."""
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
        return loss
    
    def forward_propagation(self, X):
        """
        Performs forward propagation and returns output probabilities.
        """
        # Hidden layer
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)
        
        # Output layer
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.softmax(self.output_layer_input)
        
        # (Optional debug prints; uncomment if needed)
        # print(f"Max hidden layer activation: {np.max(self.hidden_layer_output):.6f}")
        # print(f"Max output layer activation (pre-softmax): {np.max(self.output_layer_input):.6f}")
        
        return self.output_layer_output
    
    def backpropagation(self, X, y_true, y_pred):
        """
        Performs backpropagation and updates weights and biases.
        """
        m = X.shape[0]
        
        # Output layer error
        delta_output = y_pred - y_true  # (m x output_size)
        grad_weights_hidden_output = np.dot(self.hidden_layer_output.T, delta_output)
        grad_bias_output = np.sum(delta_output, axis=0, keepdims=True)
        
        # Hidden layer error
        delta_hidden = np.dot(delta_output, self.weights_hidden_output.T) * self.relu_derivative(self.hidden_layer_input)
        grad_weights_input_hidden = np.dot(X.T, delta_hidden)
        grad_bias_hidden = np.sum(delta_hidden, axis=0, keepdims=True)
        
        # Apply gradient clipping
        grad_weights_input_hidden = np.clip(grad_weights_input_hidden, -self.clip_value, self.clip_value)
        grad_weights_hidden_output = np.clip(grad_weights_hidden_output, -self.clip_value, self.clip_value)
        
        # (Optional debug prints; uncomment if needed)
        # print(f"Max weight update (input to hidden): {np.max(np.abs(grad_weights_input_hidden)):.6f}")
        # print(f"Max weight update (hidden to output): {np.max(np.abs(grad_weights_hidden_output)):.6f}")
        
        # Use layer-wise learning rates (output layer gets a smaller update)
        lr_hidden = self.learning_rate
        lr_output = self.learning_rate / 2
        
        # Update weights and biases
        self.weights_hidden_output -= lr_output * grad_weights_hidden_output
        self.bias_output -= lr_output * grad_bias_output
        self.weights_input_hidden -= lr_hidden * grad_weights_input_hidden
        self.bias_hidden -= lr_hidden * grad_bias_hidden
    
    def train(self, X, y, epochs=10, batch_size=64):
        """
        Trains the neural network using mini-batch gradient descent.
        """
        num_samples = X.shape[0]
        for epoch in range(epochs):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            epoch_loss = 0.0
            num_batches = 0
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = self.forward_propagation(X_batch)
                loss = self.cross_entropy_loss(y_batch, y_pred)
                epoch_loss += loss
                num_batches += 1
                self.backpropagation(X_batch, y_batch, y_pred)
            
            epoch_loss /= num_batches
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}")
    
    def summary(self):
        """Prints a summary of the network structure and parameter count."""
        print("Neural Network Architecture")
        print("---------------------------")
        print(f"Input Layer: {self.input_size} neurons")
        print(f"Hidden Layer: {self.hidden_size} neurons (ReLU activation)")
        print(f"Output Layer: {self.output_size} neurons (Softmax activation)")
        print(f"Total Parameters: {self.total_parameters()}")
    
    def total_parameters(self):
        """Returns the total number of trainable parameters."""
        total = (self.input_size * self.hidden_size) + self.hidden_size
        total += (self.hidden_size * self.output_size) + self.output_size
        return total

if __name__ == "__main__":
    # For testing with dummy data:
    nn = NeuralNetwork(learning_rate=0.0001, clip_value=5)
    nn.summary()
    X_dummy = np.random.rand(100, 784)
    y_dummy = np.zeros((100, 10))
    y_dummy[np.arange(100), np.random.randint(0, 10, 100)] = 1
    nn.train(X_dummy, y_dummy, epochs=5, batch_size=16)
