import numpy as np
from numpy.random import choice
from leastSquares import leastSquares
from eval_adaBoost_leastSquare import eval_adaBoost_leastSquare

def adaboostUSPS(X, Y, K, nSamples, percent):
    # Adaboost with least squares linear classifier as weak classifier on USPS data
    # for a high dimensional dataset
    #
    # INPUT:
    # X         : the dataset (numSamples x numDim)
    # Y         : labeling    (numSamples x 1)
    # K         : number of weak classifiers (scalar)
    # nSamples  : number of data points obtained by weighted sampling (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (1 x k) 
    # para      : parameters of simple classifier (K x (D+1))            
    #             For a D-dim dataset each simple classifier has D+1 parameters
    # error     : training error (1 x k)
    numSamples, numDim = np.shape(X)
    test_num = int(numSamples * percent)
    training_num = numSamples - test_num
    if training_num < nSamples:
        print("Error: 测试集错误")

    X = np.reshape(X, (numSamples, numDim))
    Y = np.reshape(Y, (numSamples, 1))
    # initialization
    W = np.ones([training_num, 1]) * 1 / training_num
    alphaK = np.zeros([K, 1])
    para = np.zeros([K, numDim + 1])
    error = np.zeros([K, 1])
    choice_test = choice(numSamples, test_num, replace=False)
    testX = X[choice_test]
    testY = Y[choice_test]
    trainingX = []
    trainingY = []
    for i in range(numSamples):
        if i in choice_test:
            pass
        else:
            trainingX.append(X[i, :])
            trainingY.append(Y[i, :])
    trainingX = np.reshape(trainingX, (training_num, numDim))
    trainingY = np.reshape(trainingY, (training_num, 1))
    for i in range(K):
        my_randomorder = choice(training_num, nSamples, replace=False)
        training_set = trainingX[my_randomorder]
        training_lables = trainingY[my_randomorder]
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
        # estimate the validation of  round i in K
        classLabels, test_result = eval_adaBoost_leastSquare(testX, alphaK[0:i, :], para[:i, :])
        error[i, 0] = sum((np.reshape(classLabels, (test_num, 1)) != testY) * 1) / test_num
    error = error[:, 0]
    #####Insert your code here for subtask 1f#####
    # Sample random a percentage of data as test data set
    return [alphaK, para, error]
