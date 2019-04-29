import numpy as np
import h5py
import math
import utils
import sys
import time
import json
from pathlib import Path
from scipy.special import softmax
from tensorflow.keras.datasets import fashion_mnist

input_folder_path = Path('../input')
output_folder_path = Path('../output')
config_folder_path = Path('./config')

def read_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
        max_iterations = config['max_iteration']
        dropout_rate = config['dropout_rate']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']
        hidden_layer_dim = config['hidden_layer_dim']
        moment_coef = config['moment_coef']
        weight_decaying = config['weight_decaying']
        non_linear = config['non_linear']
        random_seed = config['random_seed']
        return random_seed, learning_rate, max_iterations, dropout_rate, moment_coef, weight_decaying, non_linear, hidden_layer_dim, batch_size


def main():
    # Import Training Data, Training Labels and Testing Data
    # Load them from the data directory
    with h5py.File(input_folder_path / 'train_128.h5', 'r') as H:
        train_data_set = np.copy(H['data']).T
    with h5py.File(input_folder_path / 'train_label.h5', 'r') as H:
        train_labels_set = np.copy(H['label']).T
    with h5py.File(input_folder_path / 'test_128.h5', 'r') as H:
        test_data_set = np.copy(H['data']).T

    # Reshape train_labels_set to be a "m x 1" matrix so easy for validation
    train_labels_set = train_labels_set.reshape(1, train_labels_set.shape[0])
    train_data_set_shape = train_data_set.shape
    train_labels_set_shape = train_labels_set.shape
    test_data_set_shape = test_data_set.shape

    # Output the shape of the training and testing data set and training labels
    print(f'Training data set shape: {train_data_set_shape}')
    print(f'Training label set shape: {train_labels_set_shape}')
    print(f'Testing data set shape: {test_data_set_shape}')

    # Initialise hyper parameters - Default
    random_seed = 1
    learning_rate = 0.11
    max_iterations = 44
    dropout_rate = 0.95
    moment_coef=0.9
    weight_decaying=0.0007
    non_linear = 'tanh'
    hidden_layer_dim = 160
    output_layer_dim = len(set(train_labels_set[0]))
    train_size = train_data_set_shape[1]
    # train_size = 52500
    batch_size = 1500
    num_batches = int(train_size / batch_size)

    # Load hyper parameters from config
    if len(sys.argv) == 2:
        config_filename_arg = sys.argv[1]
        config_filename = config_filename_arg.split('=')[1]
        config_filename_path = config_folder_path / config_filename
        print(f'Load hyper parameters from config path: {config_filename_path}')
        random_seed, learning_rate, max_iterations, dropout_rate, moment_coef, weight_decaying, non_linear, hidden_layer_dim, batch_size = read_config(config_filename_path)

    print(f'Hyper parameters setup finished')

    # Split Train and Validation Set
    train_data, train_labels, validation_data, validation_labels = utils.train_test_split(train_data_set, train_labels_set, train_size)

    # TODO - Remove
    ((useless1, useless2), (useless3, validation_labels)) = fashion_mnist.load_data()
    validation_labels = validation_labels.reshape(1,-1)
    validation_data = test_data_set

    # Calculate Neural Network Dimensions
    layer_dims = [train_data.shape[0], hidden_layer_dim, hidden_layer_dim, output_layer_dim]
    print(f'The Multi-Layer Neural Network Dimensions: {layer_dims}')

    train_labels_encoded = utils.one_hot_encoding(train_labels, output_layer_dim)

    Acc = np.zeros((max_iterations, 1))
    start_time = int(time.time() * 1000)
    end_times = np.zeros((max_iterations, 1))

    np.random.seed(random_seed)
    parameters = utils.initialise_weights(layer_dims, type='xavier', seed=random_seed)
    print(parameters['W1'].shape)
    print(parameters['W2'].shape)
    print(parameters['W3'].shape)

    w1_new = np.zeros((layer_dims[1], layer_dims[0] + 1))
    w2_new = np.zeros((layer_dims[2], layer_dims[1] + 1))
    w3_new = np.zeros((layer_dims[3], layer_dims[2] + 1))

    momentum1 = 0
    momentum2 = 0
    momentum3 = 0

    gamma1 = np.ones((layer_dims[0], 1))
    beta1 = np.zeros((layer_dims[0], 1))
    gamma2 = np.ones((layer_dims[1], 1))
    beta2 = np.zeros((layer_dims[1], 1))
    means1 = np.zeros((layer_dims[0], num_batches))
    vars1 = np.zeros((layer_dims[0], num_batches))
    means2 = np.zeros((layer_dims[1], num_batches))
    vars2 = np.zeros((layer_dims[1], num_batches))

    beta1_new = 0
    gamma1_new = 0
    beta2_new = 0
    gamma2_new = 0

    for iteration in np.arange(0, max_iterations):
        # Shuffle for permutations
        permutations = np.random.permutation(train_size).reshape((num_batches, batch_size))
        for batch_index in np.arange(0, num_batches):
            permutation = permutations[batch_index, :]
            xtemp, xbar, xvar = utils.batch_normalisation(train_data, permutation)
            means1[:, batch_index] = xbar.reshape(1, -1)
            vars1[:, batch_index] = xvar.reshape(1, -1)
            xtemp1 = np.concatenate((np.ones((1, batch_size)), gamma1 * xtemp + beta1), axis=0)

            A1, cache = utils.forward(parameters['W1'], xtemp1, activation_function='tanh')

            A1_bar = np.mean(A1, axis=1).reshape(-1, 1)
            A1_var = np.var(A1, axis=1, ddof=1).reshape(-1, 1)
            means2[:, batch_index] = A1_bar.reshape(1, -1)
            vars2[:, batch_index] = A1_var.reshape(1, -1)

            ztemp = np.divide(A1 - A1_bar, np.sqrt(A1_var + 1e-8))
            ztemp1 = np.concatenate((np.ones((1, batch_size)), gamma2 * ztemp + beta2), axis=0)

            dropz1 = np.random.rand(layer_dims[2], layer_dims[1] + 1) < dropout_rate

            A2, cache = utils.forward(parameters['W2'] * dropz1, ztemp1, activation_function='relu')

            dropz2 = np.random.rand(layer_dims[3], layer_dims[2] + 1) < dropout_rate
            ztemp2 = np.concatenate((np.ones((1, batch_size)), A2), axis=0)
            A3 = utils.forward(parameters['W3'] * dropz2, ztemp2, activation_function='softmax')

            # Backward
            delta3 = train_labels_encoded[:, permutation] - A3

            delta2 = np.matmul(delta3.T, parameters['W3'][:,1:] * dropz2[:, 1:]) * (A2.T > 0)

            if non_linear == 'sigmoid':
                delta1 = A1 * (1 - A1) * (np.matmul(delta2, parameters['W2'][:,1:] * dropz1[:,1:]))
            elif non_linear == 'tanh':
                delta1 = (1 - np.tanh(A1.T) ** 2) * (np.matmul(delta2, parameters['W2'][:,1:] * dropz1[:,1:]))

            change3 = np.matmul(delta3, np.concatenate((np.ones((batch_size, 1)), A2.T), axis=1)) / batch_size
            change2 = np.matmul(delta2.T, ztemp1.T) / batch_size
            change1 = np.matmul(delta1.T, xtemp1.T) / batch_size

            w3_new = learning_rate * (change3 - weight_decaying * parameters['W3'] * dropz2) + moment_coef * w3_new
            w2_new = learning_rate * (change2 - weight_decaying * parameters['W2'] * dropz1) + moment_coef * w2_new
            w1_new = learning_rate * (change1 - weight_decaying * parameters['W1']) + moment_coef * w1_new

            dbeta = np.matmul(delta1, parameters['W1'][:, 1:])
            dgamma = sum(dbeta * xtemp.T) / batch_size
            dbeta = sum(dbeta) / batch_size
            gamma1_new = learning_rate * (dgamma - weight_decaying * gamma1.T) + moment_coef * gamma1_new
            beta1_new = learning_rate * (dbeta - weight_decaying * beta1.T) + moment_coef * beta1_new
            gamma1 = (gamma1.T + gamma1_new).T
            beta1 = (beta1.T + beta1_new).T
            dbeta = np.matmul(delta2, parameters['W2'][:, 1:] * dropz1[:, 1:])
            dgamma = sum(dbeta * A1.T) / batch_size
            dbeta = sum(dbeta) / batch_size
            gamma2_new = learning_rate * (dgamma - weight_decaying * gamma2.T) + moment_coef * gamma2_new
            beta2_new = learning_rate * (dbeta - weight_decaying * beta2.T) + moment_coef * beta2_new
            gamma2 = (gamma2.T + gamma2_new).T
            beta2 = (beta2.T + beta2_new).T
            #       update w2
            parameters['W3'] = parameters['W3'] + w3_new
            parameters['W2'] = parameters['W2'] + w2_new
            #       update w1
            parameters['W1'] = parameters['W1'] + w1_new


        #       batch normalisation
        xbar = np.mean(means1, axis=1).reshape(-1, 1)
        xvar = np.mean(vars1, axis=1).reshape(-1, 1) * batch_size / (batch_size - 1)
        xtest1 = np.divide((validation_data - xbar), np.sqrt(xvar + 1e-8))
        xtest2 = np.concatenate((np.ones((1, validation_data.shape[1])), gamma1 * xtest1 + beta1), axis=0)

        if non_linear == 'sigmoid':
            z1 = 1 / (1 + np.exp(-np.matmul(parameters['W1'], xtest2)))
        elif non_linear == 'tanh':
            z1 = (np.tanh(np.matmul(parameters['W1'], xtest2)))
        #     %zbar = mean(z1,2);
        #     %zvar = var(z1')';
        #     %zbar = (zbar - zbar)./sqrt(zvar+1e-8);
        zbar = np.mean(means2, axis=1).reshape(-1, 1)
        zvar = np.mean(vars2, axis=1).reshape(-1, 1) * batch_size / (batch_size - 1)
        ztemp = np.divide((z1 - zbar), np.sqrt(zvar + 1e-8))
        z11 = np.concatenate((np.ones((1, validation_data.shape[1])), gamma2 * ztemp + beta2), axis=0)
        z2 = np.matmul(parameters['W2'] * dropout_rate, z11)
        z2 = z2 * (z2 > 0)
        #     % cauculate output layer
        #     %z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
        nn = z1.shape[1]
        z3 = np.matmul(parameters['W3'] * dropout_rate, np.concatenate((np.ones((1, nn)), z2), axis=0))
        #     %for i=1:9
        #     %    o = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z']))';
        #     %end
        z3 = softmax(z3, axis=0)
        #     [~,i]=max(z3,[],2);
        res = np.argmax(z3, axis=0).reshape((1, -1))
        accuracy = np.sum(res == validation_labels) / res.shape[1] * 100
        Acc[iteration] = accuracy
        end_times[iteration] = int(time.time() * 1000) - start_time
        if iteration % 40 == 0:
            print(f'iteration: {iteration}')
            print(Acc.max(), accuracy, learning_rate)

    print(Acc.max(), learning_rate)
    print(np.where(Acc == Acc.max()))

if __name__ == "__main__":
    main()