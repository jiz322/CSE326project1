# -------------------------------------------------------------------------
'''
    Problem 4: compute sigmoid(Z), the loss function, and the gradient.
    This is the vectorized version that handle multiple training examples X.

    20/100 points
'''

import numpy as np # linear algebra
from scipy.sparse import diags
from scipy.sparse import csr_matrix

def linear(theta, X):
    '''
    theta: (n+1) x 1 column vector of model parameters
    x: (n+1) x m matrix of m training examples, each with (n+1) features.
    :return: inner product between theta and x
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return theta.T.dot(X)
    #########################################

def sigmoid(Z):
    '''
    Z: 1 x m vector. <theta, X>
    :return: A = sigmoid(Z)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return 1 / (1 + np.exp(-Z))
    #########################################

def loss(A, Y):
    '''
    A: 1 x m, sigmoid output on m training examples
    Y: 1 x m, labels of the m training examples

    You must use the sigmoid function you defined in *this* file.  ???

    :return: mean negative log-likelihood loss on m training examples.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return np.mean(-(Y*np.log(A) + (1-Y)*np.log(1-A)))
    #########################################

def dZ(Z, Y):
    '''
    Z: 1 x m vector. <theta, X>
    Y: 1 x m, label of X

    You must use the sigmoid function you defined in *this* file.

    :return: 1 x m, the gradient of the negative log-likelihood loss on all samples wrt z.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    p = sigmoid(Z) 
    return p - Y
    #########################################

def dtheta(Z, X, Y):
    '''
    Z: 1 x m vector. <theta, X>
    X: (n+1) x m, m example feature vectors.
    Y: 1 x m, label of X
    :return: d x 1, the gradient of the negative log-likelihood loss on all samples wrt w.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    phi = sigmoid(Z)
    result = np.ndarray((Y.size, Y.size))
    X = X.T
    for i in range(result.shape[0]):
        if (phi[0][i]-0.5)*(Y[0][i]-0.5) > 0:
            result[0,:] = (sigmoid(Z[0][i])-1)*X[i]
        else:
            result[1,:] = sigmoid(Z[0][i])*X[i]
    
    return np.asarray(np.mean(np.asmatrix(result).T, axis=1))
    #########################################

def Hessian(Z, X):
    '''
    Compute the Hessian matrix on m training examples.
    Z: 1 x m vector. <theta, X>
    X: (n+1) x m, m example feature vectors.
    :return: the Hessian matrix of the negative log-likelihood loss wrt theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    M = [0]*X.shape[1]

    for m in range (X.shape[1]):
        M_temp = []
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                M_temp = np.append(M_temp, float(X[i][m]*X[j][m]))
        M[m] = M_temp*sigmoid(Z[0][m])*(1-sigmoid(Z[0][m]))
    Result = np.ndarray(shape=X.shape, buffer=np.array(sum(M)))
        


    return Result
    #########################################
