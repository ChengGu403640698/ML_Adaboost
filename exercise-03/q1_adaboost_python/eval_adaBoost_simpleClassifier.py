import numpy as np


def eval_adaBoost_simpleClassifier(X, alphaK, para):
    # INPUT:
    # para	: parameters of simple classifier (K x 2) - for each round
    #           : dimension 1 is j
    #           : dimension 2 is theta
    # K         : number of classifiers used
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (numSamples x 1)
    K, _ = np.shape(para)
    numSamples, numDim = np.shape(X)
    classLabels = np.zeros([numSamples, 1])
    result = np.zeros([numSamples, 1])
    X = np.reshape(X, (numSamples, numDim))
    alphaK = np.reshape(alphaK, (K, 1))
    for i in range(K):
        j = para[i, 0]
        j = int(j)
        theta = para[i, 1]
        alpha = alphaK[i, 0]
        result += np.reshape((X[:, j] >= theta), (numSamples, 1)) * alpha
        result -= np.reshape((X[:, j] <= theta), (numSamples, 1)) * alpha
    classLabels += (result >= 0) * 1
    classLabels += (result <= 0) * -1
    classLabels = classLabels[:, 0]
    result = result[:, 0]
    #####Insert your code here for subtask 1c#####
    return classLabels, result
