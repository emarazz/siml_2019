import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Multinomial logistic regression

def softmax(z):
    '''
    Calculate the softmax function sigma: R^K -> R^K.
    '''
    sum = 1 / np.sum(np.exp(z), axis=1).reshape(-1,1)
    return np.exp(z)*sum

def compute_loss(y, tx, theta):
    '''
    Compute the loss L: R^(NxK) x R^(NxD) x R^(DxK) -> R.
    '''
    z = tx @ theta
    loss = -(z * y).sum() # sum along k and n
    loss = loss + np.log(np.exp(z).sum(axis=1)).sum()
    return loss

def compute_gradient(y, tx, theta):
    '''
    Compute the gradient of Delta_L: R^(NxK) x R^(NxD) x R^(DxK) -> R^(DxK).
    '''
    z = tx @ theta
    gradient = - tx.T @ y
    gradient = gradient + tx.T @ (softmax(z) * y)
    return gradient

def compute_acc(y, tx, theta):
    '''
    Compute the accuracy: good_predictions over all the predictions.
    '''
    z = tx @ theta
    output = np.argmax(softmax(z), axis=1)
    return np.sum(output == np.argmax(y, axis=1))/len(y)

def gradient_descent(y_train, x_train, y_test, x_test, theta_0, max_iters, alpha, print_res_each=10):
    '''
    Gradient descent.
    '''
    thetas_s, losses_train, losses_test = [], [], [] # store values to plot and display results
    theta = theta_0
    for n_iter in range(max_iters):
        loss_train = compute_loss(y_train, x_train, theta)
        loss_test = compute_loss(y_test, x_test, theta)
        losses_train.append(loss_train) # store train losses
        losses_test.append(loss_test) # store test losses
        thetas_s.append(theta) # store thetas
        if np.mod(n_iter+1,print_res_each) == 0:
            print('iter : {}/{} - train_loss = {:0.2f}, train_acc = {:0.2f}, test_loss = {:0.2f}, test_acc = {:0.2f}'.format(
                n_iter+1, max_iters ,loss_train,compute_acc(y_train, x_train, theta),
                loss_test,compute_acc(y_test, x_test, theta)))
        gradient = compute_gradient(y_train, x_train, theta) # Calculate the gradient
        theta = theta - alpha*gradient # Update the gradient
    print('')
    return losses_train, losses_test, thetas_s

def stochastic_gradient_descent(y_train, x_train, y_test, x_test, theta_0, max_iters, alpha, batch_size, print_res_each=10):
    '''
    Stochastic gradient descent.
    '''
    # Define parameters to store w and loss
    thetas_s, losses_train, losses_test = [], [], [] # store values to plot and display results
    theta = theta_0
    for n_iter in range(max_iters):
        for y_batch, x_batch in batch_iter(y_train, x_train, batch_size=batch_size):
            loss_train = compute_loss(y_train, x_train, theta)
            loss_test = compute_loss(y_test, x_test, theta)
            losses_train.append(loss_train) # store train losses
            losses_test.append(loss_test) # store test losses
            thetas_s.append(theta) # store thetas
            if np.mod(n_iter+1,print_res_each) == 0:
                print('iter : {}/{} - train_loss = {:0.2f}, train_acc = {:0.2f}, test_loss = {:0.2f}, test_acc = {:0.2f}'.format(
                    n_iter+1, max_iters ,loss_train,compute_acc(y_train, x_train, theta),
                    loss_test,compute_acc(y_test, x_test, theta)))
            gradient = compute_gradient(y_batch, x_batch, theta) # Calculate the gradient
            theta = theta - alpha*gradient # Update the gradient
    print('')
    return losses_train, losses_test, thetas_s