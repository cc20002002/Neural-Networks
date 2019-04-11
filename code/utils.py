import csv
import h5py
import numpy as np


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
    dZ = np.array(dA, copy = True)
    dZ[Z <= 0] = 0;
    return dZ;

# TODO - maybe leaky relu


def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache


def shuffle():
    return None


def split_train_test_dataset(dataset, labels, train_size=50176):
    return dataset[:, 0:train_size], dataset[:, train_size:], labels[:, 0:train_size], labels[:, train_size:]


def initialise_parameters(layer_dims, type='gauss', seed=3):
    np.random.seed(seed)
    parameters = {}
    num_layers = len(layer_dims)

    """
    Initialise Weights Matrix for Each Layer
    input dim: layer_dims[l-1]
    output dim: layer_dims[l]
    add one row due to constant vector b
    """
    for l in range(1, num_layers):
        # TODO - more testing !!!
        if type == 'gauss':
            parameters['W' + str(l)] = np.random.randn(layer_dims[l] + 1, layer_dims[l-1]) * 0.01
        if type == 'xavier':
            size = (layer_dims[l] + 1, layer_dims[l-1])
            low = - np.sqrt(6 / (layer_dims[l] + layer_dims[l-1]))
            high = np.sqrt(6 / (layer_dims[l] + layer_dims[l-1]))
            parameters['W' + str(l)] = np.random.uniform(low, high, size)
        if type == 'he':
            size = (layer_dims[l] + 1, layer_dims[l-1])
            mu = 0
            sigma = np.sqrt(2 / layer_dims[l])
            parameters['W' + str(l)] = np.random.normal(mu, sigma, size)

    return parameters

def batch_normalisation():
    return None


def forward():
    return None


def backward():
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

