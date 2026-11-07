# Week 02 - Batch Gradient Descent and SGD

## Program Title

Comparison of Batch Gradient Descent and Stochastic Gradient Descent using Neural Networks

## Aim

To implement and compare Batch Gradient Descent and Stochastic Gradient Descent for training a neural network on a binary classification dataset.

## Dataset Used

The Two-Moons dataset was generated using the `make_moons()` function from Scikit-learn.

```python
make_moons(n_samples=400, noise=0.2, random_state=1)
```

The dataset contains 400 samples with two input features and two classes.

## Program 2(a) - NumPy Implementation

A neural network was implemented from scratch using NumPy.

The implementation includes:
- ReLU activation function
- Sigmoid activation function
- Forward propagation
- Backpropagation
- Weight and bias updates
- Batch Gradient Descent
- Mini-batch SGD
- Accuracy calculation

The neural network architecture used was:

Input Layer: 2 neurons
       ↓
Hidden Layer: 16 neurons
       ↓
Hidden Layer: 16 neurons
       ↓
Output Layer: 1 neuron

## Program 2(b) - Keras Implementation

The same neural network architecture was implemented using Keras.

The model uses:
- Two hidden layers with 16 neurons each
- ReLU activation in hidden layers
- Sigmoid activation in output layer
- Binary cross-entropy loss
- SGD optimizer
- Leraning rate = 0.5
- 200 epochs

Two training configurations were compared: 
1. Batch Gradient Descent
2. Mini-batch SGD with batch size 16

## Results

The NumPy implementation successfully trained the neural network using both Batch Gradient Descent and SGD.

The Keras implementation was also trained using both batch training and mini-batch training.

The exact accuracy values may vary because of model initialization and training.

## Files

- Program-2A-BatchGD-SGD.py - Neural network implementation using NumPy
- Program-2B-Keras-BatchGD-SGD.py - Neural network implementation using Keras
- screenshots/ - Output screenshots