# stuff to use

import numpy as np

def relu(x):
    """
    computes relu activation function
    """

    x[x < 0] = 0

    return x

def relu_grad(x):
    """
    relu derivative for given input
    """
    
    grad = np.zeros_like(x)
    
    grad[x > 0] = 1.0 

    return grad

def l2_loss(y, t):
    """
    computes l2 loss between y and t
    """
    err = y - t
    loss = np.sum(np.square(err))
    return (1.0 / 2 * len(y)) * loss
