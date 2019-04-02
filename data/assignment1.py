
# coding: utf-8

# In[18]:


import numpy as np
import scipy
import h5py
import math


# In[2]:


# Import Training Data and Testing Data
with h5py.File('train_128.h5','r') as H: 
    train_data_set = np.copy(H['data'])  
with h5py.File('train_label.h5','r') as H:
    train_labels_set = np.copy(H['label'])
with h5py.File('test_128.h5', 'r') as H:
    test_data_set = np.copy(H['data'])

# Reshape train_labels_set to be a "m x 1" matrix
train_labels_set = train_labels_set.reshape(train_labels_set.shape[0], 1)
train_data_set_shape = train_data_set.shape
train_labels_set_shape = train_labels_set.shape
test_data_set_shape = test_data_set.shape

print(f'Training data set shape: {train_data_set_shape}')
print(f'Training label set shape: {train_labels_set_shape}')
print(f'Testing data set shape: {test_data_set_shape}')

learning_rate = 0.11
max_iteration = 15
droput_rate = 0.92
batch_size = 16
hidden_layer_dim = 32
output_layer_dim = len(set(train_labels_set[:,0]))
trainsize = 50048


# In[3]:


# Split Training Sample into Training Dataset and Validation Dataset
train_data = train_data_set[0:trainsize,:]
train_labels = train_labels_set[0:trainsize,:]
validation_data = train_data_set[trainsize:,:]
validation_labels = train_labels_set[trainsize:,:]

print(f'Training Data Shape: {train_data.shape}')
print(f'Training Labels Shape: {train_labels.shape}')
print(f'Validation Data Shape: {validation_data.shape}')
print(f'Validation Labels Shape: {validation_labels.shape}')


# In[149]:


# Calculate Neural Network Dimensions
layer_dims = [train_data.shape[1], hidden_layer_dim, hidden_layer_dim, output_layer_dim]
print(f'The Multi-Layer Neural Network Dimensions: {layer_dims}')


# In[44]:


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

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def one_hot_encoding(train_labels, dims):
    #y_encoded = np.zeros((dims[0], dims[1]))
    #y_encoded[np.arange(dims[0]), y[:,0]] = 1
    y == 0:max(y)
    return y_encoded
    


# In[143]:


def initialise_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    num_layers = len(layer_dims)
    
    for l in range(1, num_layers):
        # TODO - this need to be reviewed for random initialisation
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def batch_normalisation():
    return None

def forward_propagation():
    return None

def backward_propagation():
    return None


# In[ ]:


def main():
    return None


main()


# In[45]:


#y = train_labels
y = one_hot_encoding(train_labels, [train_labels.shape[0], 10])


# In[34]:


x = train_data
y = one_hot_encoding(train_labels, [train_labels.shape[0], 10])
xtest = validation_data
ytest = validation_labels

np.random.seed(3)

loss = np.arange(1, max_iteration)
w1_new = np.zeros((hidden_layer_dim, train_data.shape[1] + 1))
w2_new = np.zeros((hidden_layer_dim, hidden_layer_dim + 1))
w3_new = np.zeros((output_layer_dim, hidden_layer_dim + 1))
w1 = 2 * np.random.rand(hidden_layer_dim, train_data.shape[1] + 1) - 1
w2 = 2 * np.random.rand(hidden_layer_dim, hidden_layer_dim + 1) - 1
w3 = 2 * np.random.rand(output_layer_dim, hidden_layer_dim + 1) - 1
momentum1 = 0
momentum2 = 0
momentum3 = 0

trainsize = 50048
js = int(trainsize / batch_size)
# 16 x 128 matrix
p = np.random.randint(0, trainsize, size=trainsize).reshape((batch_size, js))
gamma1=1;
beta1=0;
gamma2=1;
beta2=0;


for iteration in np.arange(0, max_iteration):
    for j in np.arange(0, js):
        rate_drop=0.92
        
        xtemp = x[p[:,j],:]
#       batch normalisation
        xbar = np.mean(xtemp, axis=1).reshape(xtemp.shape[0], 1)
        xvar = np.var(xtemp.T, axis=0).T.reshape(xtemp.shape[0], 1)
        xtemp = np.divide((xtemp - xbar), np.sqrt(xvar + 1e-8))
        xtemp1 = np.concatenate((np.ones((batch_size,1)), gamma1*xtemp+beta1), axis=1)

        
#       calculate hidden layer
        z1 = 1 / (1 + np.exp(- np.matmul(w1, xtemp1.T))).T
#       cauculate output layer
#       z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
        drop = np.random.rand(batch_size, hidden_layer_dim) < rate_drop
        z1 = z1 * drop / rate_drop
        print(z1[0,:])
        
#         %zbar = mean(z1,2);        
#         %zvar = var(z1')';
#         %zbar = (zbar - zbar)./sqrt(zvar+1e-8);
        
        z11 = np.concatenate((np.ones((batch_size,1)), gamma2*z1+beta2), axis=1)           
        z2 = np.matmul(w2, z11.T)
        z2 = z2 * (z2 > 0)
        z2 = z2.T
        
        
        z3 = np.matmul(w3, np.concatenate((np.ones((1, batch_size)), z2.T), axis=0))
        z3 = z3.T
        z3 = softmax(z3.T).T
#       calculate gradient output layer
#       dz2/d(w2*z1)
        delta3 = y[p[:,j],:] - z3
#       calculate gragient hidden layer
        delta2 = np.matmul(delta3, w3[:,1:]) * ( z2 > 0 )
        
#       calculate gragient hidden layer
#       dz2/d(w1*xtemp1)
        delta1 = z1 * (1 - z1) * drop * (np.matmul(delta2, w2[:,1:])) / rate_drop
#       delta1 = delta1.*(delta1>0);
        change3 = np.matmul(delta3.T, np.concatenate((np.ones((batch_size,1)), z2), axis=1)) / batch_size
        change2 = np.matmul(delta2.T, np.concatenate((np.ones((batch_size,1)), z1), axis=1)) / batch_size
        change1 = np.matmul(delta1.T, xtemp1) / batch_size
#       sum of training pattern
        w3_new = learning_rate * (change3 - 0.001*w3+0.5*momentum3)
        w2_new = learning_rate * (change2 - 0.001*w2+0.5*momentum2)
        w1_new = learning_rate * (change1 - 0.001*w1+0.5*momentum1)
        momentum3 = change3
        momentum2 = change2
        momentum1 = change1
        dbeta = np.matmul(delta1, w1[:,1:])
        dgamma = sum(dbeta * xtemp) / batch_size
        dbeta = sum(dbeta) / batch_size
        gamma1 = gamma1 + 0.0005*dgamma
        beta1 = beta1 + 0.0005*dbeta
        dbeta = np.matmul(delta2, w2[:,1:])
        dgamma = sum(dbeta * z1) / batch_size
        dbeta = sum(dbeta) / batch_size
        gamma2 = gamma2 + 0.0005*dgamma
        beta2 = beta2 + 0.0005*dbeta
#       update w2
        w3 = w3 + w3_new
        w2 = w2 + w2_new
#       update w1
        w1 = w1 + w1_new
        if math.isnan(w1[0,0]):
#             print(j)
#             print(w1_new)
#             print(delta1)
#             print(delta2)
#             print(delta3)
#             print(z2)
#             print(z1)
            break   
    break
#     mean square error
#     mse(1,iteration) = sum(sum((o-t).^2)')/(numTP*numOut)
#     [~,i]=max(a,[],2);
#     
#     loss(iteration)
    print(f'iteration: {iteration}')
#     plot map and decision boundary
#     calculate hidden layer
    xbar = np.mean(xtest, axis=1).reshape(xtest.shape[0], 1)     
    xvar = np.var(xtest.T, axis=0).T.reshape(xtest.shape[0], 1) 
    xtest1 = np.divide((xtest - xbar), np.sqrt(xvar+1e-8))
    xtest2 = np.concatenate((np.ones((xtest.shape[0],1)), gamma1*xtest1+beta1), axis=1)
    z1 = 1 / (1 + np.exp(-np.matmul(w1, xtest2.T))).T
    
#     %zbar = mean(z1,2);        
#     %zvar = var(z1')';
#     %zbar = (zbar - zbar)./sqrt(zvar+1e-8);
    
    z11 = np.concatenate((np.ones((xtest.shape[0],1)), gamma2*z1+beta2), axis=1)   
    z2 = np.matmul(w2, z11.T)
    z2 = z2 * (z2>0)    
    z2 = z2.T
#     % cauculate output layer
#     %z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
    nn = z1.shape[0]
    z3 = np.matmul(w3, np.concatenate((np.ones((1,nn)), z2.T), axis=0))
    z3 = z3.T
#     %for i=1:9
#     %    o = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z']))';
#     %end
    z3 = softmax(z3.T).T 
#     [~,i]=max(z3,[],2);
    res = np.amax(z3, axis=1)

    accuracy = np.sum(res==ytest) / nn * 100
    loss[iteration] = accuracy



# In[11]:


res == ytest

