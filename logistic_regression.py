# This is logistic regression from scratch, devoid of any ML packages.

import numpy as np
import random
# scikit Used only for testing purposes
'''
from sklearn.datasets import make_classification

X, y = make_classification(n_features=2, n_redundant=0,
                           n_informative=2, random_state=13,
                           n_clusters_per_class=1)
'''


def sigmoid(x):
    # Activation function of choice
    return 1 / (1 + np.exp(-x))


def normalize(x):
    # X: an m by n matrix containing m data samples and n features for each data sample
    # Elements get gaussian normalized column by column.
    x = (x - x.mean(axis=0)) / x.std(axis=0)
    return x


def gradDesc(x, y, yCalc):
    # x: an m by n matrix containing m data samples and n features (weights) for each data sample
    # y: Ground truth result of the corresponding data samples
    # yCalc: Predicted result based on estimated slope and bias
    m = x.shape[0]
    # dw will be a column vector with n elements (the number of features)
    dw = np.dot(np.transpose(x), lossFxn(y, yCalc)) / m  # must match the dimensions of x with y for matrix operation
    db = np.sum(lossFxn(y, yCalc)) / m
    return dw, db


def lossFxn(y, yCalc):
    # Avoid division by zero
    yCalc = np.clip(yCalc, 10 ** (-15), 1 - 10 ** (-15))
    # Cross entropy
    # loss = -y * np.log(yCalc) - (1 - y) * np.log(1 - yCalc)
    # Derivative of cross entropy
    dloss = -y / yCalc + (1 - y) / (1 - yCalc)
    return dloss


def logreg_train(x, y, batchSize=50, epochs=100, learnRate=0.01):
    # x: an m by n matrix containing m data samples and n features for each data sample
    # y: Ground truth result of the corresponding data samples
    # batchSize: the number of samples used per stochastic GD iteration.
    # If batchSize == number of data samples (m) then it's just normal gradient descent
    # epochs: Number of GD iterations before termination
    # learnRate: step size for parameter update (slope and bias)

    m, n = x.shape
    if np.ndim(y) != 1:
        raise ValueError("The ground truth results must be a row or column vector")
    if np.size(y) != m:
        raise ValueError("Mismatch of number of samples and ground truth results")
    if type(batchSize) is not int or batchSize < 1:
        raise ValueError("Batch size must be a positive integer")
    y = np.reshape(y, (m, 1))  # forces the Y input into a column vector
    x = normalize(np.asarray(x))
    w = np.zeros((n, 1))  # row vector of guess for the weights for each feature
    b = 10 ** -7  # bias is assumed to be uniform across all feature

    stochasticGDReps = (m - 1) // batchSize + 1
    dwArr = np.zeros((n, stochasticGDReps * epochs))
    dbArr = np.array([])
    for i in range(epochs):
        for j in range(stochasticGDReps):  # Stochastic GD
            if batchSize >= m:
                # If batchSize >= m, it will just be regular GD
                xBatch = x
                yBatch = y
            else:
                # pick random rows of dataset as samples for stochastic GD based on the defined batch size
                rowNum = random.sample(range(0, m), batchSize)
                xBatch = x[np.asarray(rowNum), :]
                yBatch = y[np.asarray(rowNum), :]

            yCalc = sigmoid(np.dot(xBatch, w) + b)
            dw, db = gradDesc(xBatch, yBatch, yCalc)

            # Adagrad can be used too
            # w(t+1) = w(t) - learnRate(t)/sigma(t)*dw
            # learnRate decays by factor of (t+1)**(-0.5)
            # learnRate(t) = learnRate(0)/(t+1)**0.5
            # sigma(t) is the root mean square of all the previous derivatives (dwArr).
            # sigma(t) = (sum(dwArr*dwArr)/(t+1))**0.5
            # The 1/(t+1)**0.5 cancels out

            # Adagrad factors. Just delete the first `#` in each of the following four lines to use Adagrad
            # dwArr[:, i * stochasticGDReps + j] = dw[:, 0]  # Add a new column of weight gradients
            # dbArr = np.append(dbArr, db)
            w -= learnRate * dw  # / np.reshape(np.sum(dwArr * dwArr, axis=1), (n, 1)) ** 0.5
            b -= learnRate * db  # / np.sum(dbArr * dbArr) ** 0.5
    return w, b


# If directly called from a terminal, it will run based on a predetermined dataset from
# Scikit-learn. Make sure Scikit-learn is imported on the top of this file.
'''
if __name__ == '__main__':
    w, b = logreg_train(X, y, 50, 1000, 0.01)
    print('weights of their respective features: [w] = ' + str(w))
    print('calculated bias: b = ' + str(b))
'''
