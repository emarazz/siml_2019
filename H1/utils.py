import numpy as np

def standarization(x_train, x_test):
    '''
    Performs standarization, zero mean and unit variance.
    '''
    xs_train = x_train - np.mean(x_train, axis=0)
    xs_train = x_train / np.std(x_train, axis=0)
    xs_test = x_test - np.mean(x_train, axis=0)
    xs_test = x_test / np.std(x_train, axis=0)
    return xs_train, xs_test

def one_hot(y, n_classes):
    '''
    Perfoms one hot encoding.
    '''
    return np.eye(n_classes)[y-1]

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    '''
    Generate a minibatch iterator for a dataset.
    '''
    data_size = len(y)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

