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

def softmax2(z, y):
    '''
    Calculate the softmax function with the pointwise product of y: R^K x R^K -> R.
    '''
    return (softmax2 * y).sum(axis=1)

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

def compute_gradient2(y, tx, theta):
    '''
    Compute the gradient (stacked weights in a vector).
    '''
    K = theta.shape[1]
    z = tx @ theta
    tx_tilde = np.kron(np.eye(K), tx) # Kronecker product
    y_tilde = y.T.reshape(-1,1).squeeze()
    p_tilde = softmax(z).T.reshape(-1,1).squeeze()

    gradient = -tx_tilde.T @ (y_tilde - p_tilde)
    return gradient

def compute_hessian2(y, tx, theta):
    '''
    Compute the hessian (stacked weights in a vector).
    '''
    z = tx @ theta
    N = tx.shape[0]
    K = theta.shape[1]
    p = softmax(z)*y
    
    tx_tilde = np.kron(np.eye(K), tx) # Kronecker product
    W_tilde = np.zeros((N*K, N*K))
    for i in range(K):
        for j in range(K):
            if i == j:
                np.fill_diagonal(W_tilde[i*N:(i+1)*N, j*N:(j+1)*N], p[:,i]*(1-p[:,j]))
            else:
                np.fill_diagonal(W_tilde[i*N:(i+1)*N, j*N:(j+1)*N], -p[:,i]*p[:,j])

    hessian = tx_tilde.T @ W_tilde @ tx_tilde 
    return hessian

def compute_acc(y, tx, theta):
    '''
    Compute the accuracy: good_predictions over all the predictions.
    '''
    z = tx @ theta
    output = np.argmax(softmax(z), axis=1)
    return np.sum(output == np.argmax(y, axis=1))/len(y)

def gradient_descent(y_train, x_train, y_test, x_test, theta_0, max_iters, alpha, print_res_each=10, stacked=False):
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

# Stacked weights methods

def gradient_descent2(y_train, x_train, y_test, x_test, theta_0, max_iters, alpha, print_res_each=10):
    '''
    Gradient descent with stacked weights.
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
        gradient = compute_gradient2(y_train, x_train, theta) # Calculate the gradient stacked
        theta = theta.T.reshape(-1,1).squeeze() # Squeeze the weights
        theta = theta - alpha*gradient # Update the gradient
        theta = theta.reshape(theta_0.T.shape).T # Unsqueezer the weights
    print('')
    return losses_train, losses_test, thetas_s

def newtons_method2(y_train, x_train, y_test, x_test, theta_0, max_iters, alpha, print_res_each=10):
    '''
    Newtons method with stacked weights.
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
        gradient = compute_gradient2(y_train, x_train, theta) # Calculate the gradient stacked
        hessian = compute_hessian2(y_train, x_train, theta)
        theta = theta.T.reshape(-1,1).squeeze() # Squeeze the weights
        theta = theta - alpha*(np.linalg.pinv(hessian) @ gradient) # Update the gradient
        theta = theta.reshape(theta_0.T.shape).T # Unsqueezer the weights
    print('')
    return losses_train, losses_test, thetas_s

def backtracking_line_search(y_train, x_train, y_test, x_test, theta_0, rho, c, print_res_each=10):
    '''
    Newtons method with backtracking line search.
    '''
    thetas_s, losses_train, losses_test, alphas = [], [], [], [] # store values to plot and display results
    alpha, theta, n_iter, bt_cond = 1, theta_0, 0, False
    while not bt_cond:
        loss_train = compute_loss(y_train, x_train, theta) # Calculate train loss
        loss_test = compute_loss(y_test, x_test, theta) # Calculate test loss
        losses_train.append(loss_train) # store train losses
        losses_test.append(loss_test) # store test losses
        thetas_s.append(theta) # store thetas
        if np.mod(n_iter+1,print_res_each) == 0:
            print('iter : {} - train_loss = {:0.2f}, train_acc = {:0.2f}, test_loss = {:0.2f}, test_acc = {:0.2f}'.format(
                n_iter+1 ,loss_train,compute_acc(y_train, x_train, theta),
                loss_test,compute_acc(y_test, x_test, theta)))
        gradient = compute_gradient2(y_train, x_train, theta) # Calculate the gradient stacked
        hessian = compute_hessian2(y_train, x_train, theta) # Calculate Hessian
        pk = np.linalg.pinv(hessian) @ gradient
        bt_cond = compute_loss(y_train, x_train, theta + alpha*pk.T.reshape(theta.shape)) <= loss_train+alpha*c*gradient.T.dot(pk) # Test backtracking condition
        alpha *= rho # update step size
        alphas.append(alpha) # store step size
        if not bt_cond:
            theta = theta.T.reshape(-1,1).squeeze() # Squeeze the weights
            theta = theta - alpha*pk # Update the gradient
            theta = theta.reshape(theta_0.T.shape).T # Unsqueezer the weights
        n_iter += 1 # increment iteration count
    print('iterations : {} - train_loss = {:0.2f}, train_acc = {:0.2f}, test_loss = {:0.2f}, test_acc = {:0.2f}'.format(
                n_iter ,loss_train,compute_acc(y_train, x_train, theta),
                loss_test,compute_acc(y_test, x_test, theta)))
    return losses_train, losses_test, thetas_s, alphas