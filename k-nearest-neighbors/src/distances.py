import numpy as np
from math import sqrt

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    D = np.array([float(0)]*(X.shape[0]*Y.shape[0]))
    D = D.reshape(X.shape[0], Y.shape[0])
    total = 0

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for k in range(X.shape[1]):
                total += ((X[i][k]-Y[j][k])**2)
            D[i][j] = np.sqrt(total)
            total = 0
    return D

    #raise NotImplementedError()


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    D = np.array([float(0)] * (X.shape[0] * Y.shape[0]))
    D = D.reshape(X.shape[0], Y.shape[0])
    total = 0

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for k in range(X.shape[1]):
                total += (abs(X[i][k]-Y[j][k]))
            D[i][j] = total
            total = 0
    return D

    #raise NotImplementedError()


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """

    D = np.array([float(0)] * (X.shape[0] * Y.shape[0]))
    D = D.reshape(X.shape[0], Y.shape[0])
    totaltop = 0
    totalL = 0
    totalR = 0

    for i in range(X.shape[0]):
        for j in range(Y.shape[0]):
            for k in range(X.shape[1]):
                totaltop += (X[i][k]*Y[j][k])
                totalL += (X[i][k])**2
                totalR += (Y[j][k]**2)
            D[i][j] = 1 - (totaltop/(sqrt(totalL)*sqrt(totalR)))
            totaltop = 0
            totalR = 0
            totalL = 0
    return D

    # raise NotImplementedError()