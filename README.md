# MNIST Neural Network From Scratch

A neural network built entirely from scratch (using only NumPy and Matplotlib) to classify handwritten digits from the MNIST dataset. This project demonstrates the fundamentals of neural network architecture, forward propagation, backpropagation, and hyperparameter tuning—all implemented without high-level libraries like TensorFlow or PyTorch.

## Overview

This project aims to:
- **Implement a feedforward neural network** with one hidden layer.
- **Train the network** on the MNIST dataset (60,000 training images, 10,000 test images).
- **Perform hyperparameter tuning** to improve performance.
- **Visualize predictions** on a grid of test images.
- **Showcase best practices** in organizing code and using Git/GitHub for version control.

## Features

- **Data Loading & Preprocessing:**  
  Loads the raw MNIST dataset and normalizes pixel values to [0, 1]. Labels are converted to one-hot encoded vectors.

- **Neural Network Implementation:**  
  - One hidden layer (with configurable size) using ReLU activation.
  - Output layer with Softmax activation.
  - Weight initialization using He initialization.
  - Gradient clipping to stabilize training.
  
- **Training & Evaluation:**  
  - Mini-batch gradient descent.
  - Hyperparameter tuning for learning rate, hidden size, and batch size.
  - Evaluation of model accuracy on the test set.
  
- **Visualization:**  
  A grid visualization function displays test images along with their true and predicted labels.

## Installation

1. **Clone the Repository:**

   ```
   git clone https://github.com/yourusername/MNIST-NeuralNet-FromScratch.git
   cd MNIST-NeuralNet-FromScratch
   
   On windows:
   python -m venv env
   .\env\Scripts\activate
   
   On macOS/Linux:
   python3 -m venv env
   source env/bin/activate

   Install Required Packages:
   pip install numpy matplotlib

Download the MNIST Dataset:

Ensure that the following files are placed in a folder named data/ within the project directory:

train-images.idx3-ubyte
train-labels.idx1-ubyte
t10k-images.idx3-ubyte
t10k-labels.idx1-ubyte
You can download these from Yann LeCun's website or from alternative sources.

Usage
Training the Model
Run the training script:
   python train.py

This will:
Load the MNIST dataset.
Train the neural network using the selected hyperparameters.
Save the trained model weights to model_weights.npz.
Evaluate and print the test accuracy.
Display a grid visualization of test images with true and predicted labels.

Future Improvements
Additional Layers or Architectures: Experiment with deeper networks or convolutional layers.
Advanced Optimizers: Implement Adam or RMSProp for adaptive learning.
Regularization Techniques: Incorporate dropout or L2 regularization to further improve generalization.
Data Augmentation: Apply transformations (rotations, shifts) to the training data to improve robustness.


