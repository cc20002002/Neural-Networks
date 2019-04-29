import csv
import h5py
import numpy as np
from scipy.special import softmax


def one_hot_encoding(labels, dims):
    y_encoded = labels == np.arange(labels.max()+1).reshape((dims, 1))
    return y_encoded

def linear_forward(W, A):
    return np.matmul(W, A)

def linear_backward(dZ):
    return None


# Initialise Different Activation Functions
def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)


def relu(Z):
    A = np.maximum(0, Z)
    cache = Z
    return A, cache


def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache


def tanh_backward(dA, Z):
    return None


# Split Train and Test (Validation)
def train_test_split(data, labels, train_size):
    train_data = data[:, 0:train_size]
    train_labels = labels[:, 0:train_size]
    test_data = data[:, train_size:]
    test_labels = labels[:, train_size:]
    return train_data, train_labels, test_data, test_labels


def initialise_weights(layer_dims, type='default', seed=1):
    np.random.seed(seed)
    weights = {}
    num_layers = len(layer_dims)

    """
    Initialise Weights Matrix for Each Layer
    input dim: layer_dims[l-1]
    output dim: layer_dims[l]
    add one row due to constant vector b
    """
    for l in range(1, num_layers):
        if type == 'default':
            weights['W' + str(l)] = 2 * np.random.rand(layer_dims[l], layer_dims[l-1] + 1) - 1
        if type == 'gauss':
            weights['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1] + 1) * 0.01
        if type == 'xavier':
            limit = np.sqrt(6 / (layer_dims[l] + layer_dims[l-1]))
            weights['W' + str(l)] = (2 * np.random.rand(layer_dims[l], layer_dims[l-1] + 1) - 1) * limit

    return weights


# TODO - change this
def batch_normalisation(data, permutation):
    xtemp = data[:, permutation]
    xbar = np.mean(xtemp, axis=1).reshape(-1, 1)
    xvar = np.var(xtemp, axis=1, ddof=1).reshape(-1, 1)
    xtemp = np.divide(xtemp - xbar, np.sqrt(xvar + 1e-8))
    return xtemp, xbar, xvar


def forward(W, A, activation_function='tanh'):
    A_next = None
    Z = linear_forward(W, A)
    if activation_function == 'sigmoid':
        A_next = sigmoid(Z)
    elif activation_function == 'tanh':
        A_next = tanh(Z)
    elif activation_function == 'relu':
        A_next = relu(Z)
    elif activation_function == 'softmax':
        A_next = softmax(Z, axis=0)

    return A_next


def backward(dA, cache, activation_function='tanh'):
    if activation_function == 'sigmoid':
        return None
    elif activation_function == 'tanh':
        return None
    elif activation_function == 'relu':
        return None
    return None


def gradient_descent(type='batch'):
    return None


def update_parameters(layer_dims):

    return None


def predict():
    return None


def export_runlogs(filepath, data):
    with open(filepath, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(','.join(data))


def write_test_labels(test_labels):
    hf = h5py.File('../output/test_label.h5', 'w')
    hf.create_dataset('label', data=test_labels)

