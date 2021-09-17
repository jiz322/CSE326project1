# -------------------------------------------------------------------------
'''
    Problem 5: Gradient Descent and Newton method Training of Logistic Regression
    20/100 points
'''

import problem3
import problem4
from problem2 import *
import numpy as np # linear algebra
import pickle

def batch_gradient_descent(X, Y, X_test, Y_test, num_iters = 100, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    theta = np.array([[1]*(X.shape[0])]) #Y is tr_y
    Y = np.array([Y])
    Z = np.dot(theta, X)                    #X is tr_x
    A = problem4.sigmoid(Z)
    l = problem4.loss(A, Y)
    #change_rate = 0.1
    i = 0
    while i < num_iters:
        l = problem4.loss(A, Y)
        Z = np.dot(theta, X) 
        A = problem4.sigmoid(Z)
        new_theta = theta - lr* problem4.dtheta(Z, X, Y).T
    #    print(l)
    #    print("   ", problem4.dtheta(Z, X, Y)[0])
        #print("                  ", change_rate)
        # change_rate = np.mean(np.absolute(new_theta - theta))
        theta = new_theta
        i = i + 1
    Y_test = np.array([Y_test])
    Z_test = np.dot(theta, X_test)  
    A_test = problem4.sigmoid(Z_test)
    l_test = problem4.loss(A_test, Y_test)  
    norm = np.linalg.norm(theta)
    print("train lss: ", l)
    print("test lss: ", l_test)
    print("norm : ", norm)  #0.22
    return theta, (l, l_test, norm)
    #########################################

def stochastic_gradient_descent(X, Y, X_test, Y_test, num_iters = 100, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    batch_size = 2
    #X = X.T
    #np.random.shuffle(X) cannot shuffle since y is also considered. Assume data is evenly distributed
    #X = X.T
    mid = int(0.5*X.shape[1])
    batches_X = [X[:,0:mid], X[:,mid:]] #split tr_x to two batches
    Y = np.array([Y])
    batches_Y = [Y[:,0:mid], Y[:,mid:]]
    theta = np.array([[1]*(X.shape[0])]) 
    for i in range(batch_size):
        print("________batch ", i, "_____________")
        X = batches_X[i] #re-asign s with small size
        Y = batches_Y[i]          
        Z = np.dot(theta, X)                    #X is tr_x
        A = problem4.sigmoid(Z)
        l = problem4.loss(A, Y)
        #change_rate = 0.1
        i = 0
        while i < num_iters:
            l = problem4.loss(A, Y)
            Z = np.dot(theta, X) 
            A = problem4.sigmoid(Z)
            new_theta = theta - lr* problem4.dtheta(Z, X, Y).T
        #    print(l)
        #    print("   ", problem4.dtheta(Z, X, Y)[0])
            #print("                  ", change_rate)
            # change_rate = np.mean(np.absolute(new_theta - theta))
            theta = new_theta
            i = i + 1
    Y_test = np.array([Y_test])
    Z_test = np.dot(theta, X_test)  
    A_test = problem4.sigmoid(Z_test)
    l_test = problem4.loss(A_test, Y_test)  
    norm = np.linalg.norm(theta)
    print("train lss: ", l)
    print("test lss: ", l_test)
    print("norm : ", norm)  #0.22
    return theta, (l, l_test, norm)
    #########################################


def Newton_method(X, Y, X_test, Y_test, num_iters = 50, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors
    Y: 1 x m labels
    X_test: test sample vectors
    Y_test: test labels
    num_iters: number of gradient descent iterations
    log: True if you want to track the training process, by default True
    :return: (theta, training_log)
    training_log: contains training_loss, test_loss, and norm of theta
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return None, None
    #########################################


# --------------------------
def train_SGD(**kwargs):
    # use functions defined in problem3.py to perform stochastic gradient descent

    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    lr = kwargs['lr']
    log = kwargs['log']
    return stochastic_gradient_descent(tr_X, tr_y, te_X, te_y, num_iters, lr, log)


# --------------------------
def train_GD(**kwargs):
    # use functions defined in problem4.py to perform batch gradient descent

    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    lr = kwargs['lr']
    log = kwargs['log']
    return batch_gradient_descent(tr_X, tr_y, te_X, te_y, num_iters, lr, log)

# --------------------------
def train_Newton(**kwargs):
    tr_X = kwargs['Training X']
    tr_y = kwargs['Training y']
    te_X = kwargs['Test X']
    te_y = kwargs['Test y']
    num_iters = kwargs['num_iters']
    log = kwargs['log']
    return Newton_method(tr_X, tr_y, te_X, te_y, num_iters, log)


if __name__ == "__main__":
    '''
    Load and split data, and use the three training methods to train the logistic regression model.
    The training log will be recorded in three files.
    The problem5.py will be graded based on the plots in plot_training_log.ipynb (a jupyter notebook).
    You can plot the logs using the "jupyter notebook plot_training_log.ipynb" on commandline on MacOS/Linux.
    Windows should have similar functionality if you use Anaconda to manage python environments.
    '''
    X, y = loadData()
    X = appendConstant(X)
    (tr_X, tr_y), (te_X, te_y) = splitData(X, y)

    kwargs = {'Training X': tr_X,
              'Training y': tr_y,
              'Test X': te_X,
              'Test y': te_y,
              'num_iters': 1000,
              'lr': 0.01,
              'log': True}

    theta, training_log = train_SGD(**kwargs)
    with open('./data/SGD_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)


    theta, training_log = train_GD(**kwargs)
    with open('./data/batch_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)
#
#
    theta, training_log = train_Newton(**kwargs)
    with open('./data/newton_outcome.pkl', 'wb') as f:
        pickle.dump((theta, training_log), f)


