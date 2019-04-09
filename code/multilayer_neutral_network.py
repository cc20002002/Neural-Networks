
import numpy as np
from scipy.special import softmax
import h5py
import math


def predict_test_dataset():
    return None


def write_test_labels():
    return None


def main():
    # Import Training Data, Training Labels and Testing Data
    # Load them from the data directory
    with h5py.File('../data/train_128.h5', 'r') as H:
        train_data_set = np.copy(H['data'])
    with h5py.File('../data/train_label.h5', 'r') as H:
        train_labels_set = np.copy(H['label'])
    with h5py.File('../data/test_128.h5', 'r') as H:
        test_data_set = np.copy(H['data'])

    # Reshape train_labels_set to be a "m x 1" matrix so easy for validation
    train_labels_set = train_labels_set.reshape(train_labels_set.shape[0], 1)
    train_data_set_shape = train_data_set.shape
    train_labels_set_shape = train_labels_set.shape
    test_data_set_shape = test_data_set.shape

    print(f'Training data set shape: {train_data_set_shape}')
    print(f'Training label set shape: {train_labels_set_shape}')
    print(f'Testing data set shape: {test_data_set_shape}')


main()