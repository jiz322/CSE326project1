# -------------------------------------------------------------------------
'''
    Problem 3: compute sigmoid(<theta, x>), the loss function, and the gradient.
    This is the single training example version.

    20/100 points
'''

import numpy as np # linear algebra
import math

def linear(theta, x):
    '''
    theta: (n+1) x 1 column vector of model parameters
    x: (n+1) x 1 column vector of an example features. Must be a sparse csc_matrix
    :return: inner product between theta and x
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return np.dot(theta,x)
    #########################################

def sigmoid(z):
    '''
    z: scalar. <theta, x>
    :return: sigmoid(z)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return 1 / (1 + np.exp(-z))
    #########################################

def loss(a, y): #Lecture Note logistic regression 19
    '''
    a: 1 x 1, sigmoid of an example x
    y: {0,1}, the label of the corresponding example x
    :return: negative log-likelihood loss on (x, y).
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    #negative a for sigmoid result in 1-a
    return -(y*math.log(a) + (1-y)*math.log(1-a))
    #########################################

def dz(z, y): #bu kao pu probabaly hw2 q1
    '''
    z: scalar. <theta, x>
    y: {0,1}, label of x
    :return: the gradient of the negative log-likelihood loss on (x, y) wrt z.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    p = sigmoid(z) 
    return p-y #网上瞎找的
    #########################################

def dtheta(z, x, y):  #Lecture Note logistic regression 42
    '''
    z: scalar. <theta, x>
    x: (n+1) x 1 vector, an example feature vector
    y: {0,1}, label of x
    :return: the gradient of the negative log-likelihood loss on (x, y) wrt theta.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    
    phi = sigmoid(z)
    result = 0
    if (phi-0.5)*(y-0.5) > 0:
        result = (sigmoid(z)-1)*x
    else:
        result = sigmoid(z)*x
    return result
    #########################################

def Hessian(z, x): #Lecture Note logistic regression between 25&26
    '''
    C;ompute the Hessian matrix on a single training example.
    z: scalar. <theta, x>
    x: (n+1) x 1 vector, an example feature vector
    :return: the Hessian matrix of the negative log-likelihood loss wrt theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    MX = []
    for i in range(x.size):
        for j in range(x.size):
            MX.append(float(x[0][i]*x[0][j]))
    M = np.ndarray(shape=(x.size,x.size), buffer=np.array(MX))
    return sigmoid(z)*(1-sigmoid(z))*M
    #########################################    
