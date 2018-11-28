import numpy as np


def lin_Classifier(X, Y, numSamples):
    X = np.reshape(X, (1, numSamples))
    Y = np.reshape(Y, (1, numSamples))
    theta_temp = 0
    error = 0
    X_temp = np.ones([2, numSamples])
    X_temp[1, :] = X[0, :]
    W = np.dot(np.linalg.inv(np.dot(X_temp, X_temp.T)), X_temp)
    W = np.dot(W, Y.T)
    theta = - W[0, 0]
    p = W[1, 0]
    training_result = np.ones([numSamples, 1])
    for i in range(numSamples):
        if(p * X[0, i] < theta):
            training_result[i, 0] *= -1
        if(training_result[i, 0] != Y[0, i]):
            error = error + 1

    return theta, error



def simpleClassifier(X, Y):
    # Select a simple classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    #
    # OUTPUT:
    # theta 	: threshold value for the decision (scalar)
    # j 		: the dimension to "look at" (scalar)
    #initialiazation
    numSamples, numDim = np.shape(X)
    min_error = numSamples
    theta = 0
    j = 0
    for i in range(numDim):
        theta_temp, error = lin_Classifier(X[:, i], Y, numSamples)
        if error < min_error:
            min_error = error
            theta = theta_temp
            j = i
    #####Insert your code here for subtask 1b#####
    return j, theta
