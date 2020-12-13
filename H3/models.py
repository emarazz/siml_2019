import numpy as np
from scipy.special import factorial
from utils import *

def normal_d(x,mu,sigma):
    """
    normal distribution
    """
    pdf = 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-1/(2*np.power(sigma,2))*np.power(x-mu,2))
    return pdf

def gamma_d(x,a,b):
    """
    gamma distribution pdf
    """
    gamma = factorial(a-1)
    pdf = 1/gamma* np.power(b,a) * np.power(x,a-1) * np.exp(-b*x)
    pdf[np.isnan(pdf)] = 0
    return pdf

def dirichlet_d(x,delta):
    """
    dirichlet distribution pdf
    """
    return np.repeat(np.prod(np.power(x,delta-1)),len(delta))

def p_zi_xi(x,rho,mu,phi):
    """
    posterior probability that the observation xi has been generated from the k-th component
    """
    x = x.reshape(-1,1)
    rho = rho.reshape(1,-1)
    mu = mu.reshape(1,-1)
    phi = phi.reshape(1,-1)

    numerator = rho*np.sqrt(phi) * np.exp(-phi/2 * np.power(x-mu,2))
    denominator = numerator.sum(axis=1).reshape(-1,1)
    
    return numerator/denominator


# def dirichlet_d2(x,alpha):
#     """
#     dirichlet distribution
#     """
#     gamma_num = np.array([factorial(alpha[i]) for i in range(len(alpha))])
#     gamma_den = np.factorial(np.sum(alpha)-1)
#     beta = np.prod(gamma_num)/gamma_den
#     return 1/beta * np.prod(np.power(x,alpha-1))

# def normal_d(x,mu,sigma):
#     """
#     multivariate normal distribution pdf
#     """
#     x = np.array(x)
#     r = len(x)
    
#     pdf = 1/( np.sqrt(np.power(2*np.pi,r) * np.linalg.det(sigma)))
#     pdf = pdf * np.exp(-1/2 * (x-mu).reshape(1,-1) @ np.linalg.inv(sigma) @ (x-mu).reshape(-1,1))
#     return pdf.squeeze()
