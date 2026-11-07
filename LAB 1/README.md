# Week 01 - Neural Network Implementation for XOR

## Program Title

Implementation of XOR using a Neural Network

## Aim

To implement the XOR logical operation using a neural network and compare a manually implemented neural network with a Keras-based neural network.

## Dataset Used

The XOR truth table was used as the dataset.

| Input 1 | Input 2 | Expected Output |
|---------:|---------:|----------------:|
| 0        | 0        | 0               |
| 0        | 1        | 1               |
| 1        | 0        | 1               |
| 1        | 1        | 0               |

## Program 1(a)

A neural network was implemented from scratch using NumPy.

The implementation includes:

- Sigmoid activation function
- Sigmoid derivative
- Forward propagation
- Error calculation
- Backpropagation
- Weight and bias updates
- 10,000 training epochs

## Program 1(b)

A neural network was implemented using Keras.

The model consists of:

- Input layer with 2 input features
- Hidden layer with 8 neurons and ReLU activation
- Output layer with 1 neuron and Sigmoid activation
- Binary cross-entropy loss
- SGD optimizer
- Learning rate of 0.1
- 1000 training epochs

## Results

Program 1(a) successfully learned the XOR operation and produced predictions close to:

```text
0
1
1
0
```

Program 1(b) achieved high classification accuracy and produced predictions corresponding to the XOR truth table.

## Files

- Program-1A-XOR-Neural-Network.py - Manual neural network implementation using NumPy
- Program-1B-Keras-XOR.py - Neural network implementation using Keras
- screenshots/ - Screenshots of the program outputs
