# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from sklearn.decomposition import IncrementalPCA as PCA
from tensorflow.keras.datasets import fashion_mnist
import h5py
import numpy as np
from matplotlib import pyplot as plt
((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
with h5py.File('train_128.h5','r') as H: 
    train_data_set = np.copy(H['data'])  
with h5py.File('train_label.h5','r') as H:
    train_labels_set = np.copy(H['label'])
with h5py.File('test_128.h5', 'r') as H:
    test_data_set = np.copy(H['data'])

sum(train_labels_set==trainY)



'''
pca = PCA(n_components=128) 
pca.fit(trainX.reshape(60000,-1)) 
trainX2=pca.transform(trainX.reshape(60000,-1))
train_data_set2=pca.inverse_transform(train_data_set.reshape(60000,-1))
train_data_set2=train_data_set2.reshape(60000,28,28)
example=train_data_set2[503,:,:].squeeze()

plt.imshow(example)

Xtr2=pca.transform(trainX) 
