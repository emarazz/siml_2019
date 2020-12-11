import numpy as np
from utils import *

def normal_d(x,mu,sigma):
    """
    normal distribution
    """
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/2 * np.power((x-mu)/sigma,2))

def gamma_d(x,alpha,beta):
    """
    gamma distribution
    """
    gamma = factorial(alpha-1)
    return np.power(beta,alpha)/gamma * np.power(x,alpha-1) * np.exp(-beta*x)

def dirichlet_d(x,alpha):
    """
    dirichlet distribution
    """
    gamma_num = np.array([factorial(alpha[i]) for i in range(len(alpha))])
    gamma_den = np.factorial(np.sum(alpha)-1)
    beta = np.prod(gamma_num)/gamma_den
    return 1/beta * np.prod(np.power(x,alpha-1))


