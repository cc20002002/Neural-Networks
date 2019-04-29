import numpy as np
import h5py
import math
import sys
import time
import json
import argparse
import utils
from pathlib import Path
from scipy.special import softmax

input_folder_path = Path('../input')
output_folder_path = Path('../output')
config_folder_path = Path('./config')


def main():
    # Import Data Sets
    train_data_set, train_labels_set = utils.load_data()
    test_data_set = None
    test_labels = None

    # Load hyper parameters from config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='hyperparameters config')
    parser.add_argument('--test_data', help='dataset for test, used for prediction')
    parser.add_argument('--test_labels', help='labels for test, used for test accuracy')

    args = parser.parse_args()

    if args.config is not None:
        config_filename_path = config_folder_path / args.config
        hyperparameters = utils.init_hyperparameters(config_filename_path)

    if args.test_data is not None:
        test_data_set_filepath = input_folder_path / args.test_data
        test_data_set = utils.load_h5(test_data_set_filepath)
        test_data_set_shape = test_data_set.shape
        print(f'Testing data set shape: {test_data_set_shape}')

    if args.test_labels is not None:
        test_labels_filepath = input_folder_path / args.test_labels
        test_labels = utils.load_h5(test_labels_filepath, 'label')
        test_data_set_shape = test_data_set.shape
        print(f'Testing labels shape: {test_data_set_shape}')

    output_layer_dim = len(set(train_labels_set[:, 0]))
    train_size = train_data_set.shape[0]
    num_batches = int(train_size / hyperparameters['batch_size'])

    # Setup train data for experiment
    train_data = train_data_set[0:train_size, :]
    train_labels = train_labels_set[0:train_size, :]
    train_labels = utils.one_hot_encoding(train_labels, output_layer_dim)
    train_data_sample_size, train_data_feature_size = train_data.shape

    # Setup seeds for random
    np.random.seed(hyperparameters['random_seed'])

    # Dimensions for different layers - easier for weights initialisation in neural network
    layer_dims = [train_data_feature_size, hyperparameters['hidden_layer_dim'], hyperparameters['hidden_layer_dim'], output_layer_dim]

    # Initialise weights and parameters for neural network
    # Weights for propagation and backpropagation
    weights = utils.initialise_weights(layer_dims)
    # Parameters for Batch normalisation, Gradient descent
    parameters = utils.initialise_parameters(layer_dims, num_batches)

    weights, parameters = utils.model_fit(train_data=train_data, train_labels=train_labels, test_data=test_data_set, test_labels=test_labels, weights=weights, parameters=parameters, hyperparameters=hyperparameters, layer_dims=layer_dims, train_size=train_size, num_batches=num_batches)

    # Export the predicted labels to h5 file into the output folder
    if test_data_set is not None:
        predicted_labels = utils.predict(test_data_set, weights, parameters, hyperparameters)
        utils.export_predicted_labels(predicted_labels)

if __name__ == "__main__":
    main()