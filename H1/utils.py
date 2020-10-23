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

