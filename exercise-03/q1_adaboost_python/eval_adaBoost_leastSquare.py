import numpy as np

def eval_adaBoost_leastSquare(X, alphaK, para):
    # INPUT:
    # para		: parameters of simple classifier (K x (D +1)) 
    #           : dimension 1 is w0
    #           : dimension 2 is w1
    #           : dimension 3 is w2
    #             and so on
    # K         : number of classifiers used
    # X         : test data points (numSamples x numDim)
    #
    # OUTPUT:
    # classLabels: labels for data points (numSamples x 1)
    # result     : weighted sum of all the K classifier (scalar)

    K, _ = np.shape(para)
    numSamples, numDim = np.shape(X)
    classLabels = np.zeros([numSamples, 1])
    result = np.zeros([numSamples, 1])
    X = np.reshape(X, (numSamples, numDim))
    alphaK = np.reshape(alphaK, (K, 1))
    for i in range(K):
        bias = para[i, 0]
        weight = para[i, 1:]
        alpha = alphaK[i, 0]
        result += np.reshape((np.dot(X, np.reshape(weight, (numDim, 1))) + bias >= 0), (numSamples, 1)) * alpha
        result -= np.reshape((np.dot(X, np.reshape(weight, (numDim, 1))) + bias <= 0), (numSamples, 1)) * alpha
    classLabels += (result >= 0) * 1
    classLabels += (result <= 0) * -1
    classLabels = classLabels[:, 0]
    result = result[:, 0]
    #####Insert your code here for subtask 1e#####

    return [classLabels, result]

