import numpy as np

def factorial(x):
    """
    factorial
    """
    output = 1
    for i in range(1,x+1):
        output = output*i
    return output