# stuff to use

import numpy as np

def sigmoid(x):
    """
    computes sigmoid activation function
    """
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_grad(x):
    """
    computes sigmoid activation function gradient
    """
    return sigmoid(x) * (1.0 - sigmoid(x))

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
    n = len(y)
    return loss / (2.0 * n)

def l2_loss_grad(y, t):
    """
    computes gradient of l2 loss with respect to predictions y
    """

    n = len(y)

    return (y - t) / n
