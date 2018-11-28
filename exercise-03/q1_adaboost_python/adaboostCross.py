import numpy as np
from numpy.random import choice
from simpleClassifier import simpleClassifier
from eval_adaBoost_simpleClassifier import eval_adaBoost_simpleClassifier

def adaboostCross(X, Y, K, nSamples, percent):
    # Adaboost with an additional cross validation routine
    #
    # INPUT:
    # X         : training examples (numSamples x numDims )
    # Y         : training lables (numSamples x 1)
    # K         : number of weak classifiers to select (scalar)
    #             (the _maximal_ iteration count - possibly abort earlier)
    # nSamples  : number of training examples which are selected in each round. (scalar)
    #             The sampling needs to be weighted!
    # percent   : percentage of the data set that is used as test data set (scalar)
    #
    # OUTPUT:
    # alphaK    : voting weights (K x 1)
    # para      : parameters of simple classifier (K x 2)
    # testX     : test dataset (numTestSamples x numDim)
    # testY     : test labels  (numTestSamples x 1)
    # error	    : error rate on validation set after each of the K iterations (K x 1)
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
    para = np.zeros([K, 2])
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
    trainingX = np.reshape(trainingX, (training_num, 2))
    trainingY = np.reshape(trainingY, (training_num, 1))
    for i in range(K):
        my_randomorder = choice(training_num, nSamples, replace=False)
        training_set = trainingX[my_randomorder]
        training_lables = trainingY[my_randomorder]
        W_nsample = W[my_randomorder]
        result_lables = np.ones([nSamples, 1])
        j, theta = simpleClassifier(training_set, training_lables)
        para[i, 0] = j
        para[i, 1] = theta
        result_lables[training_set[:, j] < theta] = -1
        error_temp = W_nsample[result_lables != training_lables]
        training_error = sum(error_temp) / sum(W_nsample)
        alphaK[i, 0] = 0.5 * np.log((1 - training_error) / training_error)
        W_nsample = W_nsample * (result_lables == training_lables
                                 ) + W_nsample * (result_lables != training_lables) * np.exp(alphaK[i, 0])
        W[my_randomorder] = W_nsample
        #estimate the validation of  round i in K
        classLabels, test_result = eval_adaBoost_simpleClassifier(testX, alphaK[0:i, :], para[:i, :])
        error[i, 0] = sum((np.reshape(classLabels, (test_num, 1)) != testY) * 1) / test_num
    error = error[:, 0]
    #####Insert your code here for subtask 1d#####
    # Randomly sample a percentage of the data as test data set
    return alphaK, para, testX, testY, error

