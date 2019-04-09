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

# TODO - maybe leaky relu


def tanh(Z):
    A = np.tanh(Z)
    cache = Z
    return A, cache


def shuffle():
    return None


def split_train_test_dataset(dataset, labels, train_size=50176):
    return dataset[:, 0:train_size], dataset[:, train_size:], labels[:, 0:train_size], labels[:, train_size:]


def initialise_parameters(layer_dims, type='gauss'):
    np.random.seed(3)
    parameters = {}
    num_layers = len(layer_dims)

    for l in range(1, num_layers):
        # TODO - more testing !!!
        if type == 'gauss':
            parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        if type == 'xavier':
            size = (layer_dims[l], layer_dims[l-1])
            low = - np.sqrt(6 / (layer_dims[l] + layer_dims[l-1]))
            high = np.sqrt(6 / (layer_dims[l] + layer_dims[l-1]))
            parameters['W' + str(l)] = np.random.uniform(low, high, size)
        if type == 'he':
            size = (layer_dims[l], layer_dims[l-1])
            mu = 0
            sigma = np.sqrt(2 / layer_dims[l])
            parameters['W' + str(l)] = np.random.normal(mu, sigma, size)

    return parameters


def forward():
    return None


def backward():
    return None


def gradient_descent(type='batch'):
    return None


def update_parameters(layer_dims):

    return None