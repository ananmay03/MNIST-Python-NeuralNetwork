import numpy as np
from data import load_mnist_dataset
from network import NeuralNetwork
import matplotlib.pyplot as plt

def one_hot_encode(y, num_classes=10):
    m = y.shape[0]
    one_hot = np.zeros((m, num_classes))
    one_hot[np.arange(m), y] = 1
    return one_hot

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
    return accuracy

if __name__ == "__main__":
    # Load MNIST dataset
    (X_train, y_train), (X_test, y_test) = load_mnist_dataset()
    y_train = one_hot_encode(y_train)
    y_test = one_hot_encode(y_test)

    ### Option 1: Train with fixed hyperparameters
    # Uncomment the following lines to do a single training run with fixed hyperparameters.
    # nn = NeuralNetwork(learning_rate=0.0001, clip_value=5)
    # print("\nTraining on MNIST dataset with fixed hyperparameters...")
    # nn.train(X_train, y_train, epochs=10, batch_size=64)
    # evaluate_model(nn, X_test, y_test)
    
    ### Option 2: Hyperparameter Tuning via Grid Search
    print("Starting Hyperparameter Tuning...\n")
    
    # Define hyperparameters to test
    learning_rates = [0.00005, 0.0001, 0.0002]
    hidden_sizes = [128, 256]
    batch_sizes = [32, 64]

    results = {}

    for lr in learning_rates:
        for hs in hidden_sizes:
            for bs in batch_sizes:
                print(f"\nTraining with learning_rate={lr}, hidden_size={hs}, batch_size={bs}")
                nn = NeuralNetwork(hidden_size=hs, learning_rate=lr, clip_value=5)
                # Train for a few epochs (e.g., 5 epochs for quick tuning)
                nn.train(X_train, y_train, epochs=5, batch_size=bs)
                acc = evaluate_model(nn, X_test, y_test)
                results[(lr, hs, bs)] = acc

    print("\nSummary of Hyperparameter Tuning:")
    for params, acc in results.items():
        print(f"Learning Rate: {params[0]}, Hidden Size: {params[1]}, Batch Size: {params[2]} -> Accuracy: {acc * 100:.2f}%")
