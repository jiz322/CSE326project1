# -------------------------------------------------------------------------
'''
    Problem 2: reading data set from a file, and then split them into training, validation and test sets.

    The functions for handling data

    20/100 points
'''

import numpy as np # for linear algebra

def loadData():
    '''
        Read all labeled examples from the text files.
        Note that the data/X.txt has a row for a feature vector for intelligibility.

        n: number of features
        m: number of examples.

        :return: X: numpy.ndarray. Shape = [n, m]
                y: numpy.ndarray. Shape = [m, ]
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    data = np.loadtxt("./data/X.txt")
    X = data.transpose()
    y = np.loadtxt("./data/y.txt")
    #########################################
    return X, y


def appendConstant(X):
    '''
    Appending constant "1" to the beginning of each training feature vector.
    X: numpy.ndarray. Shape = [n, m]
    :return: return the training samples with the appended 1. Shape = [n+1, m]
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    ones = np.ones((1, X.shape[1]))
    return np.concatenate((ones, X), axis=0)
    #########################################


def splitData(X, y, train_ratio = 0.8):
    '''
	X: numpy.ndarray. Shape = [n+1, m]
	y: numpy.ndarray. Shape = [m, ]
    split_ratio: the ratio of examples go into the Training, Validation, and Test sets.
    Split the whole dataset into Training, Validation, and Test sets.
    :return: return (training_X, training_y), (test_X, test_y).
            training_X is a (n+1, m_tr) matrix with m_tr training examples;
            training_y is a (m_tr, ) column vector;
            test_X is a (n+1, m_test) matrix with m_test test examples;
            training_y is a (m_test, ) column vector.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    tr_X = X[:, 0:int(0.8*X.shape[1])]
    tr_y = y[0:int(0.8*X.shape[1])]
    test_X = X[:, int(0.8*X.shape[1]):]
    test_y = y[int(0.8*X.shape[1]):]
    return (tr_X, tr_y), (test_X, test_y)
    #########################################

