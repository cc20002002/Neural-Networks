import h5py
import json
import numpy as np
from pathlib import Path
from scipy.special import softmax

input_folder_path = Path('../input')
output_folder_path = Path('../output')
config_folder_path = Path('./config')


def load_h5(filepath, dataname='data'):
    with h5py.File(filepath, 'r') as H:
        data = np.copy(H[dataname])
        return data


def load_data():
    # Import Training Data, Training Labels and Testing Data
    # Load them from the input directory
    print(f'Start loading training data, training labels and test data set')

    train_data_set_filepath = (input_folder_path / 'train_128.h5')
    train_labels_filepath = (input_folder_path / 'train_label.h5')
    # test_data_set_filepath = (input_folder_path / 'test_128.h5')
    train_data_set = load_h5(train_data_set_filepath)
    train_labels_set = load_h5(train_labels_filepath, 'label')
    # test_data_set = load_h5(test_data_set_filepath)

    # Reshape train_labels_set to be a "m x 1" matrix so easy for validation
    train_labels_set = train_labels_set.reshape(-1, 1)
    train_data_set_shape = train_data_set.shape
    train_labels_set_shape = train_labels_set.shape
    # test_data_set_shape = test_data_set.shape

    # Output the shape of the training and testing data set and training labels
    print(f'Training data set shape: {train_data_set_shape}')
    print(f'Training label set shape: {train_labels_set_shape}')
    # print(f'Testing data set shape: {test_data_set_shape}')
    return train_data_set, train_labels_set


def init_hyperparameters(config_path=None):
    # Initialise hyper parameters - Default
    random_seed = 1
    learning_rate = 0.11
    max_iterations = 44
    dropout_rate = 0.95
    moment_coef = 0.9
    weight_decaying = 0.0007
    non_linear = 'tanh'
    hidden_layer_dim = 160
    batch_size = 1500

    if config_path is not None:
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
            print(f'Loaded hyperparameters from config path: {config_path}')
    else:
        print(f'Use default hyperparameters')

    hyperparameters = {
        'random_seed': random_seed,
        'learning_rate': learning_rate,
        'max_iterations': max_iterations,
        'dropout_rate': dropout_rate,
        'moment_coef': moment_coef,
        'weight_decaying': weight_decaying,
        'non_linear': non_linear,
        'hidden_layer_dim': hidden_layer_dim,
        'batch_size': batch_size
    }

    print(f'hyperparameters are: {hyperparameters}')

    return hyperparameters


def export_predicted_labels(labels):
    output_file_path = output_folder_path / 'predicted_labels.h5'
    hf = h5py.File(output_file_path, 'w')
    hf.create_dataset('label', data=labels)
    print(f'Successfully exported the predicted_labels into {output_file_path}')


def one_hot_encoding(train_labels, dims):
    y_encoded = train_labels == np.arange(train_labels.max() + 1).reshape((1, dims))
    return y_encoded


# Xavier initialisation for weights
def initialise_weights(layer_dims):
    weights = {}
    num_layers = len(layer_dims)

    for l in range(1, num_layers):
        limit = np.sqrt(6 / (layer_dims[l] + layer_dims[l-1]))
        weights[f'w{l}'] = (2 * np.random.rand(layer_dims[l], layer_dims[l-1] + 1) - 1) * limit
        weights[f'w{l}_new'] = np.zeros((layer_dims[l], layer_dims[l-1] + 1))
    return weights


# Initialise parameters
def initialise_parameters(layer_dims, num_batches):
    parameters = {}
    parameters['gamma1'] = np.ones((1, layer_dims[0]));
    parameters['beta1'] = np.zeros((1, layer_dims[0]));
    parameters['gamma2'] = np.ones((1, layer_dims[1]));
    parameters['beta2'] = np.zeros((1, layer_dims[1]));
    parameters['means1'] = np.zeros((num_batches, layer_dims[0]))
    parameters['vars1'] = np.zeros((num_batches, layer_dims[0]))
    parameters['means2'] = np.zeros((num_batches, layer_dims[1]))
    parameters['vars2'] = np.zeros((num_batches, layer_dims[1]))
    parameters['beta1_new'] = 0
    parameters['gamma1_new'] = 0
    parameters['beta2_new'] = 0
    parameters['gamma2_new'] = 0
    return parameters


# Batch Normalisation
def batch_normalisation(data):
    means = np.mean(data, axis=0).reshape(1, -1)
    vars = np.var(data, axis=0, ddof=1).reshape(1, -1)
    norm_data = np.divide(data - means, np.sqrt(vars + 1e-8))
    return norm_data, means, vars


# Linear Activation Forward
def linear_activation_forward(W, A, activation='tanh'):
    if activation == 'tanh':
        return np.tanh(np.matmul(W, A))
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(- np.matmul(W, A)))
    elif activation == 'relu':
        Z = np.matmul(W, A)
        return Z * (Z > 0)
    else:
        return np.matmul(W, A)


# Three layers forward
def layers_forward(weights, data, parameters, layer_dims, hyperparameters):
    minibatch_norm_data, means1, vars1 = batch_normalisation(data)

    b1 = np.ones((hyperparameters['batch_size'], 1))
    norm_A1 = np.concatenate((b1, parameters['gamma1'] * minibatch_norm_data + parameters['beta1']), axis=1)
    A2 = linear_activation_forward(weights['w1'], norm_A1.T, activation=hyperparameters['non_linear']).T

    norm_A2, means2, vars2 = batch_normalisation(A2)

    b2 = np.ones((hyperparameters['batch_size'], 1))
    norm_A2 = np.concatenate((b2, parameters['gamma2'] * norm_A2 + parameters['beta2']), axis=1)
    drop_A2 = np.random.rand(layer_dims[2], layer_dims[1] + 1) < hyperparameters['dropout_rate']
    A3 = linear_activation_forward(weights['w2'] * drop_A2, norm_A2.T, activation='relu').T

    drop_A3 = np.random.rand(layer_dims[3], layer_dims[2] + 1) < hyperparameters['dropout_rate']
    b3 = np.ones((1, hyperparameters['batch_size']))

    A4 = linear_activation_forward(weights['w3'] * drop_A3, np.concatenate((b3, A3.T), axis=0), activation='linear').T

    caches = {}
    caches['minibatch_norm_data'] = minibatch_norm_data
    caches['means1'] = means1
    caches['vars1'] = vars1
    caches['means2'] = means2
    caches['vars2'] = vars2
    caches['norm_A1'] = norm_A1
    caches['A2'] = A2
    caches['norm_A2'] = norm_A2
    caches['drop_A2'] = drop_A2
    caches['A3'] = A3
    caches['drop_A3'] = drop_A3

    return A4, caches


# Three layers backward for Gradient Descent
def layers_backward(exp_labels, pred_labels, weights, parameters, caches, hyperparameters):
    delta3 = exp_labels - pred_labels
    delta2 = np.matmul(delta3, weights['w3'][:, 1:] * caches['drop_A3'][:, 1:]) * (caches['A3'] > 0)
    if hyperparameters['non_linear'] == 'sigmoid':
        delta1 = caches['A2'] * (1 - caches['A2']) * (np.matmul(delta2, weights['w2'][:, 1:] * caches['drop_A2'][:, 1:]))
    elif hyperparameters['non_linear'] == 'tanh':
        delta1 = (1 - np.tanh(caches['A2']) ** 2) * (np.matmul(delta2, weights['w2'][:, 1:] * caches['drop_A2'][:, 1:]))
    change3 = np.matmul(delta3.T, np.concatenate((np.ones((hyperparameters['batch_size'], 1)), caches['A3']), axis=1)) / hyperparameters['batch_size']
    change2 = np.matmul(delta2.T, caches['norm_A2']) / hyperparameters['batch_size']
    change1 = np.matmul(delta1.T, caches['norm_A1']) / hyperparameters['batch_size']
    dbeta = np.matmul(delta1, weights['w1'][:, 1:])
    dgamma = sum(dbeta * caches['minibatch_norm_data']) / hyperparameters['batch_size']
    dbeta = sum(dbeta) / hyperparameters['batch_size']
    parameters['gamma1_new'] = hyperparameters['learning_rate'] * (dgamma - hyperparameters['weight_decaying'] * parameters['gamma1']) + hyperparameters['moment_coef'] * parameters['gamma1_new']
    parameters['beta1_new'] = hyperparameters['learning_rate'] * (dbeta - hyperparameters['weight_decaying'] * parameters['beta1']) + hyperparameters['moment_coef'] * parameters['beta1_new']
    parameters['gamma1'] = parameters['gamma1'] + parameters['gamma1_new']
    parameters['beta1'] = parameters['beta1'] + parameters['beta1_new']
    dbeta = np.matmul(delta2, weights['w2'][:, 1:] * caches['drop_A2'][:, 1:])
    dgamma = sum(dbeta * caches['A2']) / hyperparameters['batch_size']
    dbeta = sum(dbeta) / hyperparameters['batch_size']
    parameters['gamma2_new'] = hyperparameters['learning_rate'] * (dgamma - hyperparameters['weight_decaying'] * parameters['gamma2']) + hyperparameters['moment_coef'] * parameters['gamma2_new']
    parameters['beta2_new'] = hyperparameters['learning_rate'] * (dbeta - hyperparameters['weight_decaying'] * parameters['beta2']) + hyperparameters['moment_coef'] * parameters['beta2_new']
    parameters['gamma2'] = parameters['gamma2'] + parameters['gamma2_new']
    parameters['beta2'] = parameters['beta2'] + parameters['beta2_new']
    return change1, change2, change3


def update_weights(weights, hyperparameters, caches, change1, change2, change3):
    weights['w3_new'] = hyperparameters['learning_rate'] * (change3 - hyperparameters['weight_decaying'] * weights['w3'] * caches['drop_A3']) + hyperparameters['moment_coef'] * weights['w3_new']
    weights['w2_new'] = hyperparameters['learning_rate'] * (change2 - hyperparameters['weight_decaying'] * weights['w2'] * caches['drop_A2']) + hyperparameters['moment_coef'] * weights['w2_new']
    weights['w1_new'] = hyperparameters['learning_rate'] * (change1 - hyperparameters['weight_decaying'] * weights['w1']) + hyperparameters['moment_coef'] * weights['w1_new']
    weights['w3'] = weights['w3'] + weights['w3_new']
    weights['w2'] = weights['w2'] + weights['w2_new']
    weights['w1'] = weights['w1'] + weights['w1_new']
    return weights


# Fit the model
def model_fit(train_data, train_labels, test_data, test_labels, weights, parameters, hyperparameters, layer_dims, train_size, num_batches):
    accuracy = np.zeros((hyperparameters['max_iterations'], 1))

    print(f'Model Fit - Training iterations')
    # Start iterations for training
    for iteration in np.arange(0, hyperparameters['max_iterations'] + 1):
        # Permutations for Mini-Batch
        p = np.random.permutation(train_size).reshape((num_batches, hyperparameters['batch_size'])).T

        # Mini-Batch
        for j in np.arange(0, num_batches):

            # Batch normalisation
            minibatch_data = train_data[p[:, j], :]

            # Multi-Layers Forward
            A4, caches = layers_forward(weights, minibatch_data, parameters, layer_dims, hyperparameters)
            A4 = softmax(A4, axis=1)
            parameters['means1'][j, :] = caches['means1']
            parameters['vars1'][j, :] = caches['vars1']
            parameters['means2'][j, :] = caches['means2']
            parameters['vars2'][j, :] = caches['vars2']

            # Mutli-Layers Backward
            exp_labels = train_labels[p[:, j], :]
            pred_labels = A4
            change1, change2, change3 = layers_backward(exp_labels, pred_labels, weights, parameters, caches, hyperparameters)

            weights = update_weights(weights, hyperparameters, caches, change1, change2, change3)

        # Cross Validation if validation set provided, otherwise print training accuracy
        if test_labels is not None:
            acc = evaluate_acc(test_data, test_labels, weights, parameters, hyperparameters)
            accuracy[iteration-1] = acc
            print(f'Epoch: {iteration}, \t Test Accuracy {round(accuracy.max(),2)}%')
        else:
            train_labels_orig = np.where(train_labels == True)[1].reshape(-1)
            acc = evaluate_acc(train_data, train_labels_orig, weights, parameters, hyperparameters)
            accuracy[iteration-1] = acc
            print(f'Epoch: {iteration}, \t Training Accuracy {round(accuracy.max(),2)}%')

    print(f'Complete model fitting. The output file is saved at ... ')
    return weights, parameters


def evaluate_acc(test_data, test_labels, weights, parameters, hyperparameters):
    predicted_labels = predict(test_data, weights, parameters, hyperparameters)
    acc = np.sum(predicted_labels == test_labels) / predicted_labels.shape[0] * 100
    return acc


def predict(data, weights, parameters, hyperparameters):
    means1 = np.mean(parameters['means1'], axis=0).reshape(1, -1)
    vars1 = np.mean(parameters['vars1'], axis=0).reshape(1, -1) * hyperparameters['batch_size'] / (hyperparameters['batch_size'] - 1)
    norm_validation_data = np.divide((data - means1), np.sqrt(vars1 + 1e-8))
    norm_validation_data = np.concatenate((np.ones((data.shape[0], 1)), parameters['gamma1'] * norm_validation_data + parameters['beta1']),axis=1)
    A2 = linear_activation_forward(weights['w1'], norm_validation_data.T, activation=hyperparameters['non_linear']).T

    means2 = np.mean(parameters['means2'], axis=0).reshape(1, -1)
    vars2 = np.mean(parameters['vars2'], axis=0).reshape(1, -1) * hyperparameters['batch_size'] / (hyperparameters['batch_size'] - 1)
    norm_A2 = np.divide((A2 - means2), np.sqrt(vars2 + 1e-8))
    norm_A2 = np.concatenate((np.ones((data.shape[0], 1)), parameters['gamma2'] * norm_A2 + parameters['beta2']), axis=1)
    A3 = linear_activation_forward(weights['w2'] * hyperparameters['dropout_rate'], norm_A2.T, activation='relu').T

    A4 = np.matmul(weights['w3'] * hyperparameters['dropout_rate'], np.concatenate((np.ones((1, data.shape[0])), A3.T), axis=0)).T
    A4 = softmax(A4, axis=1)

    predicted_labels = np.argmax(A4, axis=1).T
    return predicted_labels
