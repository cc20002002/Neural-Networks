
import numpy as np
from scipy.special import softmax
import h5py
import math
import utils

def predict_test_dataset():
    return None


def write_test_labels():
    return None


def main():
    # Import Training Data, Training Labels and Testing Data
    # Load them from the data directory
    with h5py.File('../data/train_128.h5', 'r') as H:
        train_data_set = np.copy(H['data']).T
    with h5py.File('../data/train_label.h5', 'r') as H:
        train_labels_set = np.copy(H['label']).T
    with h5py.File('../data/test_128.h5', 'r') as H:
        test_data_set = np.copy(H['data']).T

    # Reshape train_labels_set to be a "m x 1" matrix so easy for validation
    train_labels_set = train_labels_set.reshape(1, train_labels_set.shape[0])
    train_data_set_shape = train_data_set.shape
    train_labels_set_shape = train_labels_set.shape
    test_data_set_shape = test_data_set.shape

    print(f'Training data set shape: {train_data_set_shape}')
    print(f'Training label set shape: {train_labels_set_shape}')
    print(f'Testing data set shape: {test_data_set_shape}')

    # Split the training and testing dataset
    # TODO - Input Shuffle
    print(f'Split training and test data')
    train_data, test_data, train_labels, test_labels = utils.split_train_test_dataset(train_data_set, train_labels_set)

    hidden_layer_dim = 161
    output_layer_dim = 1
    num_iterations = 400

    """
    Three Layer Neural Networks
    Two Hidden Layers + One Output Layer
    So the layer dimension will be:
        1. training data dimension (axis=0)
        2. first hidden layer dimension (axis=0)
        3. second hidden layer dimension (axis=0)
        4. output layer dimension (axis=0)
    Vertical Dimensions of Each Layer (axis=0)
    """
    layer_dims = [train_data.shape[0], hidden_layer_dim, hidden_layer_dim, output_layer_dim]
    print(f'The Multi-Layer Neural Network Dimensions: {layer_dims}')

    for i in range(0, num_iterations):
        print(i)

main()