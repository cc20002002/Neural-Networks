##########
##########
##########  TODO - This file is used to compare test labels acc for submission, please do not add this into submission please
##########
##########

import h5py
import utils
import numpy as np
from pathlib import Path

from tensorflow.keras.datasets import fashion_mnist
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()

output_folder_path = Path('../output')

output_file_path = output_folder_path / 'predicted_labels.h5'
train_labels_set = utils.load_h5(output_file_path, 'label')

print(testY.shape)
print(train_labels_set.shape)

acc = np.sum(testY == train_labels_set) / train_labels_set.shape[0] * 100
print(f'acc is {acc}, check this if ok, if so lets submit')