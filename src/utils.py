import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def row_normalize(matrix, eps=1e-12):
    row_sums = matrix.sum(axis=1, keepdims=True)
    return matrix / (eps + row_sums)