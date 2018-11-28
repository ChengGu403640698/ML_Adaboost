import numpy as np
from numpy.random import choice
from leastSquares import leastSquares

def adaboostLSLC(X, Y, K, nSamples):
    # Adaboost with least squares linear classifier as weak classifier
    # for a D-dim dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (iteration number of Adaboost) (scalar)
    # nSamples  : number of data which are weighted sampled (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of least square classifier (K x 3) 
    #             For a D-dim dataset each least square classifier has D+1 parameters
    #             w0, w1, w2........wD
    numSamples, numDim = np.shape(X)
    X = np.reshape(X, (numSamples, numDim))
    Y = np.reshape(Y, (numSamples, 1))
    W = np.ones([numSamples, 1]) * 1 / numSamples
    alphaK = np.zeros([K, 1])
    para = np.zeros([K, numDim + 1])
    for i in range(K):
        my_randomorder = choice(numSamples, nSamples, replace=False)
        training_set = X[my_randomorder]
        training_lables = Y[my_randomorder]
        W_nsample = W[my_randomorder]
        result_lables = np.ones([nSamples, 1])
        weight, bias = leastSquares(training_set, training_lables)
        para[i, 0] = bias
        para[i, 1:] = np.reshape(weight, (1, numDim))
        result_lables[np.dot(training_set, para[i, 1:]) + bias <= 0] = -1
        error_temp = W_nsample[result_lables != training_lables]
        if np.size(error_temp) == 0 or sum(error_temp) == 0:
            alphaK[i, 0] = 100
        else:
            error = sum(error_temp) / sum(W_nsample)
            alphaK[i, 0] = 0.5 * np.log((1 - error) / error)
            W_nsample = W_nsample * (result_lables == training_lables
                                     ) + W_nsample * (result_lables != training_lables) * np.exp(alphaK[i, 0])
        W[my_randomorder] = W_nsample
    #####Insert your code here for subtask 1e#####

    return [alphaK, para]
