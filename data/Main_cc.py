
# coding: utf-8

# In[18]:

import csv
import sys
import time
import json
import numpy as np
from scipy.special import softmax
import h5py
import math
import IPython


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
max_iteration = 44
dropout_rate = 0.95
batch_size = 1500
hidden_layer_dim = 160
output_layer_dim = len(set(train_labels_set[:,0]))
trainsize = 60000#50176 train it hard
moment_coef=0.9
weight_decaying=0.0007

non_linear='sigmoid'
non_linear='tanh'
'''tanh 89.02 23 iterations
learning_rate = 0.005
max_iteration = 400
dropout_rate = .8
batch_size = 1000
hidden_layer_dim = 900
output_layer_dim = len(set(train_labels_set[:,0]))
trainsize = 60000#50176 train it hard
moment_coef=0.9
weight_decaying=0.0015
'''
'''sigmoid 89.6 400 iterations
learning_rate = 0.11
max_iteration = 400
dropout_rate = 1
batch_size = 1000
hidden_layer_dim = 150
output_layer_dim = len(set(train_labels_set[:,0]))
trainsize = 60000#50176 train it hard
moment_coef=0.88
weight_decaying=0.0005
'''

# In[3]:

if len(sys.argv) == 2:
    print(f'Using the config in this run')
    filepath_arg = sys.argv[1]
    filepath = filepath_arg.split('=')[1]
    with open(filepath, 'r') as f:
        config = json.load(f)
        max_iteration = config['max_iteration']
        dropout_rate = config['dropout_rate']
        learning_rate = config['learning_rate']
        batch_size = config['batch_size']
        hidden_layer_dim = config['hidden_layer_dim']
        moment_coef = config['moment_coef']
        weight_decaying = config['weight_decaying']
        non_linear = config['non_linear']


# Split Training Sample into Training Dataset and Validation Dataset
train_data = train_data_set[0:trainsize,:]
train_labels = train_labels_set[0:trainsize,:]
validation_data = train_data_set[0:trainsize,:]
validation_labels = train_labels_set[0:trainsize,:]

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

def dsigmoid_dA():
    dfdA=1
    return dfdA

def drelu_dA():
    dfdA=1
    return dfdA

def dtanh_dA():
    dfdA=1
    return dfdA

def one_hot_encoding(train_labels, dims):
    #vectorization half the time taken
    #t = time.time()
    y_encoded = train_labels==np.arange(train_labels.max()+1).reshape((1, dims))
    #print( time.time()-t)
    
    #t = time.time()
    #y_encoded = np.zeros((dims[0], dims[1]))
    #y_encoded[np.arange(dims[0]), train_labels[:,0]] = 1
    #print( time.time()-t)
    return y_encoded
    


# In[143]:
def sgd_momentum(df_dx, x0, conf_para=None):
    # this is just an example of how to shape you sgd function
    if conf_para is None:
        conf_para = {}
    
    conf_para.setdefault('n_iter', 1000) #number of iterations
    conf_para.setdefault('learning_rate', 0.001) #learning rate
    conf_para.setdefault('momentum', 0.9) #momentum parameter
    
    x_traj = []
    x_traj.append(x0)
    v = np.zeros_like(x0)
    
    for iter in range(1, conf_para['n_iter']+1):
        dfdx = np.array(df_dx(x_traj[-1][0], x_traj[-1][1]))
        v = conf_para['momentum']*v - conf_para['learning_rate']*dfdx
        x_traj.append(x_traj[-1]+v)    
    return x_traj

def adam(df_dx, x0, conf_para=None):
    # try to use adam in the algorithm as well. please check weight decay
    # as I have not tested. I tested the function without weight decay
    if conf_para is None:
        conf_para = {}
    
    conf_para.setdefault('n_iter', 1000) #number of iterations
    conf_para.setdefault('learning_rate', 0.001) #learning rate
    conf_para.setdefault('beta1', 0.85)
    conf_para.setdefault('beta2', 0.999)
    conf_para.setdefault('v', 0.)
    conf_para.setdefault('m', 0.)
    conf_para.setdefault('epsilon', 1e-8)
    x_traj = []
    x_traj.append(x0)
    v=conf_para['v']
    m=conf_para['m']
    for iter in range(1, conf_para['n_iter']+1):
      dfdx = np.array(df_dx(x_traj[-1][0], x_traj[-1][1]))
      v = conf_para['beta2']*v + (1-conf_para['beta2'])*dfdx**2
      m = conf_para['beta1']*m + (1-conf_para['beta1'])*dfdx
      mhat = m/(1-conf_para['beta1']**iter)
      vhat = v/(1-conf_para['beta2']**iter)      
      x_traj.append(x_traj[-1]-conf_para['learning_rate']/np.sqrt(vhat+conf_para['epsilon'])*mhat)-weight_decaying*x_traj[-1] #
    
    return x_traj

def initialise_parameters(layer_dims):
    np.random.seed(3)
    parameters = {}
    num_layers = len(layer_dims)
    
    for l in range(1, num_layers):
        # TODO - this need to be reviewed for random initialisation
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def batch_normalisation(x,mean,var):
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
y = one_hot_encoding(train_labels,  10)

# In[34]:


x = train_data
numTP, numFV =x.shape
xtest = validation_data
ytest = validation_labels
#todo delete this before submission
from tensorflow.keras.datasets import fashion_mnist
((trainX, trainY), (xtest, ytest)) = fashion_mnist.load_data()
ytest=ytest.reshape(-1,1)
print(ytest)
xtest=test_data_set



Acc = np.zeros(( max_iteration,1)) 
Loss = np.zeros(( max_iteration,1)) 

#todo: shape is wrong

w1_new = np.zeros((hidden_layer_dim, train_data.shape[1] + 1))
w2_new = np.zeros((hidden_layer_dim, hidden_layer_dim + 1))
w3_new = np.zeros((output_layer_dim, hidden_layer_dim + 1))
np.random.seed(1)

# TODO Comment out this original initialisation for now
# Changed to N(0,1)
llmit=np.sqrt(6/(hidden_layer_dim+train_data.shape[1]))
w1 = (2 * np.random.rand(hidden_layer_dim, train_data.shape[1] + 1) - 1)*llmit
#w1[:,0]=0
llmit=np.sqrt(6/(hidden_layer_dim+hidden_layer_dim))
w2 = (2 * np.random.rand(hidden_layer_dim, hidden_layer_dim + 1) - 1)*llmit
#w2[:,0]=0
llmit=np.sqrt(6/(output_layer_dim+hidden_layer_dim))
w3 = (2 * np.random.rand(output_layer_dim, hidden_layer_dim + 1) - 1)*llmit
#w3[:,0]=0


#w1 = 0.01 * np.random.rand(hidden_layer_dim, train_data.shape[1] + 1)
#w2 = 0.01 * np.random.rand(hidden_layer_dim, hidden_layer_dim + 1)
#w3 = 0.01 * np.random.rand(output_layer_dim, hidden_layer_dim + 1)
momentum1 = 0
momentum2 = 0
momentum3 = 0


batches = int(np.floor(trainsize / batch_size))
gamma1=np.ones((1,numFV));
beta1=np.zeros((1,numFV));
gamma2=np.ones((1,hidden_layer_dim));
beta2=np.zeros((1,hidden_layer_dim));
means1=np.zeros((batches,numFV));
vars1=np.zeros((batches,numFV));
means2=np.zeros((batches,hidden_layer_dim));
vars2=np.zeros((batches,hidden_layer_dim));
# 16 x 128 matrix
#p = np.arange(trainsize).reshape((batch_size, batches))


beta1_new=0
gamma1_new=0
beta2_new=0
gamma2_new=0

start_time = int(time.time() * 1000)

for iteration in np.arange(0, max_iteration):
    p = np.random.permutation(trainsize).reshape((batches,batch_size)).T
    for j in np.arange(0, batches):
        
        xtemp = x[p[:,j],:]
#       batch normalisation
        xbar = np.mean(xtemp, axis=0).reshape(1,-1)
        xvar = np.var(xtemp, axis=0,ddof=1).reshape(1,-1)
        xtemp = np.divide(xtemp - xbar, np.sqrt(xvar + 1e-8))        
        means1[j,:]=xbar;
        vars1[j,:]=xvar;
        xtemp1 = np.concatenate((np.ones((batch_size,1)), gamma1*xtemp+beta1), axis=1)
        
#       calculate hidden layer
        #possible overflow todo
        # sigmoid
        if non_linear=='sigmoid':
            z1 = 1 / (1 + np.exp(- np.matmul(w1, xtemp1.T))).T
        # tanh
        elif non_linear=='tanh':
            z1 =  (np.tanh(np.matmul(w1, xtemp1.T))).T
#       cauculate output layer
#       z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
        #np.random.seed(3)
        
        #z1 = z1 * drop / dropout_rate
        
        
        ##
#       batch normalisation        
        zbar = np.mean(z1,axis=0).reshape(1,-1)        
        zvar = np.var(z1, axis=0,ddof=1).reshape(1,-1)
        means2[j,:]=zbar
        vars2[j,:]=zvar
        ztemp = (z1 - zbar)/np.sqrt(zvar+1e-8);            
        ztemp1 = np.concatenate((np.ones((batch_size,1)), gamma2*ztemp+beta2), axis=1)           
        
        dropz1 = np.random.rand(hidden_layer_dim,hidden_layer_dim+1) < dropout_rate
        
        z2 = np.matmul(w2*dropz1, ztemp1.T)
        z2 = z2 * (z2 > 0)
        z2 = z2.T
        
        ##
        dropz2 = np.random.rand(output_layer_dim,hidden_layer_dim+1) < dropout_rate
        z3 = np.matmul(w3*dropz2, np.concatenate((np.ones((1, batch_size)), z2.T), axis=0))
        z3 = z3.T
        
        z3 = softmax(z3,axis=1)
        

 #       calculate gradient output layer
 #       dz2/d(w2*z1)
        delta3 = y[p[:,j],:] - z3
         #       calculate gragient hidden layer
        delta2 = np.matmul(delta3, w3[:,1:]*dropz2[:,1:]) * ( z2 > 0 )
        
         #       calculate gragient hidden layer
         #       dz2/d(w1*xtemp1)
        #sigmoid
        if non_linear=='sigmoid':
            delta1 = z1 * (1 - z1) * (np.matmul(delta2, w2[:,1:]*dropz1[:,1:])) 
        #tanh
        elif non_linear=='tanh':
            delta1 = (1 - np.tanh(z1)**2) * (np.matmul(delta2, w2[:,1:]*dropz1[:,1:])) 
         #       delta1 = delta1.*(delta1>0);
        change3 = np.matmul(delta3.T, np.concatenate((np.ones((batch_size,1)), z2), axis=1)) / batch_size
        change2 = np.matmul(delta2.T, ztemp1) / batch_size
        change1 = np.matmul(delta1.T, xtemp1) / batch_size
         #       sum of training pattern
        w3_new = learning_rate * (change3 - weight_decaying*w3*dropz2)+moment_coef*w3_new
        w2_new = learning_rate * (change2 - weight_decaying*w2*dropz1)+moment_coef*w2_new
        w1_new = learning_rate * (change1 - weight_decaying*w1)+moment_coef*w1_new
        
        dbeta = np.matmul(delta1, w1[:,1:])
        dgamma = sum(dbeta * xtemp) / batch_size
        dbeta = sum(dbeta) / batch_size
        gamma1_new = learning_rate*(dgamma - weight_decaying*gamma1)+moment_coef*gamma1_new
        beta1_new = learning_rate*(dbeta - weight_decaying*beta1)+moment_coef*beta1_new
        gamma1 = gamma1 + gamma1_new
        beta1 = beta1 + beta1_new
        dbeta = np.matmul(delta2, w2[:,1:]*dropz1[:,1:])
        dgamma = sum(dbeta * z1) / batch_size
        dbeta = sum(dbeta) / batch_size
        gamma2_new = learning_rate*(dgamma - weight_decaying*gamma2)+moment_coef*gamma2_new
        beta2_new = learning_rate*(dbeta - weight_decaying*beta2)+moment_coef*beta2_new
        gamma2 = gamma2 + gamma2_new
        beta2 = beta2 + beta2_new
        #       update w2
        w3 = w3 + w3_new
        w2 = w2 + w2_new
        #       update w1
        w1 = w1 + w1_new
        if math.isnan(w1[0,0]):
          print(j)
          print(iteration)
          print(w1_new)
          print(delta1)
          print(delta2)
          print(delta3)
          print(z2)
          print(z1)
          IPython.embed()
         #     mean square error
         #     mse(1,iteration) = sum(sum((o-t).^2)')/(numTP*numOut)
         #     [~,i]=max(a,[],2);
         #
         #     Acc(iteration)
    
         #     plot map and decision boundary
         #     calculate hidden layer
#       batch normalisation
    xbar = np.mean(means1, axis=0).reshape(1, -1)
    xvar = np.mean(vars1, axis=0).reshape(1, -1)*batch_size/(batch_size-1)
    xtest1 = np.divide((xtest - xbar), np.sqrt(xvar+1e-8))
    xtest2 = np.concatenate((np.ones((xtest.shape[0],1)), gamma1*xtest1+beta1), axis=1)
    if non_linear=='sigmoid':
        z1 = 1 / (1 + np.exp(-np.matmul(w1, xtest2.T))).T
    elif non_linear=='tanh':
        z1 =  (np.tanh(np.matmul(w1, xtest2.T))).T
     #     %zbar = mean(z1,2);
     #     %zvar = var(z1')';
     #     %zbar = (zbar - zbar)./sqrt(zvar+1e-8);
    
    zbar = np.mean(means2, axis=0).reshape(1, -1)
    zvar = np.mean(vars2, axis=0).reshape(1, -1)*batch_size/(batch_size-1)
    ztemp = np.divide((z1 - zbar), np.sqrt(zvar+1e-8))
    z11 = np.concatenate((np.ones((xtest.shape[0],1)), gamma2*ztemp+beta2), axis=1)
    z2 = np.matmul(w2*dropout_rate, z11.T)
    z2 = z2 * (z2>0)
    z2 = z2.T
     #     % cauculate output layer
     #     %z1 = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z2']))';
    nn = z1.shape[0]
    z3 = np.matmul(w3*dropout_rate, np.concatenate((np.ones((1,nn)), z2.T), axis=0))
    z3 = z3.T
     #     %for i=1:9
     #     %    o = 1 ./ (1 + exp(-w2 * [ones(1,numTP); z']))';
     #     %end
    z3 = softmax(z3,axis=1)
     #     [~,i]=max(z3,[],2);
    res = np.argmax(z3, axis=1).T.reshape((-1,1))
    
    accuracy = np.sum(res==ytest) / res.shape[0] * 100
    
    Acc[iteration] = accuracy
    if iteration %40 ==0:
        print(f'iteration: {iteration}')
        print(Acc.max(),accuracy,learning_rate)
    # TODO - need to revisit early stopping
    # This is for debugging, turn it off
    # if accuracy > 85:
    #     break
    
     # In[11]:
    
        
res == ytest
        

# Predict the test labels
# TODO - conver to function !!!!!!!!!!
print(f'***************************************************************')
print(f'After the optimisation:')
print(f'w1.shape is : {w1.shape}')
print(f'w2.shape is : {w2.shape}')
print(f'w3.shape is : {w3.shape}')

xbar = np.mean(means1, axis=0).reshape(1, -1)
xvar = np.mean(vars1, axis=0).reshape(1, -1)*batch_size/(batch_size-1)
t_input = np.divide((test_data_set - xbar), np.sqrt(xvar+1e-8))
t_input = np.concatenate((np.ones((test_data_set.shape[0],1)), gamma1*t_input+beta1), axis=1)
# sigmoid
if non_linear == 'sigmoid':
    z1_output = 1 / (1 + np.exp(- np.matmul(w1, t_input.T))).T
# tanh
elif non_linear == 'tanh':
    z1_output = (np.tanh(np.matmul(w1, t_input.T))).T
print(z1_output.shape)


zbar = np.mean(means2, axis=0).reshape(1, -1)
zvar = np.mean(vars2, axis=0).reshape(1, -1)*batch_size/(batch_size-1)
ztemp = np.divide((z1_output - zbar), np.sqrt(zvar+1e-8))
z2 = np.concatenate((np.ones((test_data_set.shape[0],1)), gamma2*ztemp+beta2), axis=1)
z2_output = np.matmul(w2 * dropout_rate, z2.T).T
z2_output = z2_output * (z2_output>0)
print(z2_output.shape)

z3 = np.concatenate((np.ones((test_data_set.shape[0], 1)), z2_output), axis=1)
z3_output = np.matmul(w3 * dropout_rate, z3.T).T
print(z3_output.shape)

test_actuals = softmax(z3_output, axis=1)
print(test_actuals.shape)
test_actuals = np.argmax(test_actuals, axis=1)

print(test_actuals)


matched_sum = sum(test_actuals==ytest)

hf = h5py.File('../output/predicted_labels.h5', 'w')
hf.create_dataset('label', data=test_actuals)

print(f'***************************************************************')

end_time = int(time.time() * 1000)


def export_runlogs(filepath, data):
    fieldnames = ['Id', 'Runtime', 'Accurarcy', 'Activation function type', 'Batch Normalisation', 'Weight decay rate', 'Momentum rate', 'Dropout rate', 'Learning rate', 'np.argmax']
    with open(filepath, 'a+', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data)

output_filepath = '../output/runlogs.csv'
num_lines = sum(1 for line in open(output_filepath))
run_time = end_time - start_time
max_acc = Acc.max()
npargmax = np.where(Acc == Acc.max())[0]
job_status = {
    'Id': num_lines,
    'Runtime': run_time,
    'Accurarcy': str(max_acc),
    'Activation function type': non_linear,
    'Batch Normalisation': 'True',
    'Weight decay rate': weight_decaying,
    'Momentum rate': moment_coef,
    'Dropout rate': dropout_rate,
    'Learning rate': learning_rate,
    'np.argmax': npargmax
}
export_runlogs('../output/runlogs.csv', job_status)
