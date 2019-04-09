import numpy as np


# Initialise Different Activation Functions
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache


def shuffle():
    return None


def split_train_test_dataset(dataset, labels, train_size=50176):
    return dataset[:, 0:train_size], dataset[:, train_size:], labels[:, 0:train_size], labels[:, train_size:]


def initialise_parameters():
    return None


def forward():
    return None


def backward():
    return None


def gradient_descent(type='batch'):
    return None


def update_parameters():
    return None