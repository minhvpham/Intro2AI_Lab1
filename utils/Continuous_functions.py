import numpy as np
import matplotlib.pyplot as plt


def rastrigin(X):
    """
    Rastrigin function (minimization). Global minimum is 0.0 at X = [0, 0,...].
    X is a NumPy array (vector) of shape (n_dims,).
    """
    A = 10
    n_dims = len(X)
    return A * n_dims + np.sum(X**2 - A * np.cos(2 * np.pi * X), axis=0)