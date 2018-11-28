import numpy as np
from numpy.random import choice

from simpleClassifier import simpleClassifier
def adaboostSimple(X, Y, K, nSamples):
    # Adaboost with decision stump classifier as weak classifier
    #
    # INPUT:
    # X         : training examples (numSamples x numDim)
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar) 
    #             (the _maximal_ iteration count - possibly abort earlier
    #              when error is zero)
    # nSamples  : number of training examples which are selected in each round (scalar)
    #             The sampling needs to be weighted!
    #             Hint - look at the function 'choice' in package numpy.random
    #
    # OUTPUT:
    # alphaK 	: voting weights (K x 1) - for each round
    # para		: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    numSamples, numDim = np.shape(X)
    X = np.reshape(X, (numSamples, numDim))
    Y = np.reshape(Y, (numSamples, 1))
    W = np.ones([numSamples, 1]) * 1 / numSamples
    alphaK = np.zeros([K, 1])
    para = np.zeros([K, 2])
    for i in range(K):
        my_randomorder = choice(numSamples, nSamples, replace=False)
        training_set = X[my_randomorder]
        training_lables = Y[my_randomorder]
        W_nsample = W[my_randomorder]
        result_lables = np.ones([nSamples, 1])
        j, theta = simpleClassifier(training_set, training_lables)
        para[i, 0] = j
        para[i, 1] = theta
        result_lables[training_set[:, j] < theta] = -1
        error_temp = W_nsample[result_lables != training_lables]
        error = sum(error_temp) / sum(W_nsample)
        alphaK[i, 0] = 0.5 * np.log((1 - error) / error)
        W_nsample = W_nsample * (result_lables == training_lables
                                 ) + W_nsample * (result_lables != training_lables) * np.exp(alphaK[i, 0])
        W[my_randomorder] = W_nsample
    #####Insert your code here for subtask 1c#####
    return alphaK, para
