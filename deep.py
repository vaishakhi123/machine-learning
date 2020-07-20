#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist

def relu(z):
    return np.maximum(0, z)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def flatten(x_train, y_train):
    # mnist dataset comes in the shape (m, x, y), without color
    # with color stuff can be just flattened with a * 3 at the end
    x = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))/255
    y = (y_train)/255
    return x.T, y.T

def cost(Y, A, m):
    J = (-1/m) * (np.dot(Y, np.log(A)) + np.dot(1 - Y, np.log(1 - A)))  
    return J

def diffSigmoid(z):
    return np.dot(sigmoid(z), (1-sigmoid(z)))

def diffRelu(z):
    return relu(z)/z

def initParams(layers): 
    params = {}
    for i in range(0, len(layers) - 1):
        params["W" + str(i+1)] = np.random.randn(layers[i+1], layers[i]) * 0.1 # Ws are independent of m
        params["b" + str(i+1)] = np.zeros((layers[i+1], 1)) # b gets broadcasted to remove dependence on m for later use
    return params

def updateParams(params, backwardCache, layers, alpha):
    for i in range(1, len(layers)):
        params["W" + str(i)] -= alpha * backwardCache["dW" + str(i)]
        params["b" + str(i)] -= alpha * backwardCache["db" + str(i)]
    return params

def forwardProp(X, params, layers):
    forwardCache = {}
    forwardCache["A0"] = X
    for i in range(1, len(layers)):
        forwardCache["Z" + str(i)] = np.dot(params["W" + str(i)], forwardCache["A" + str(i-1)]) + params["b" + str(i)]
        forwardCache["A" + str(i)] = relu(forwardCache["Z" + str(i)]) # using relu for all activations
    return forwardCache

def backwardProp(X, Y, params, forwardCache, layers):
    m = X.shape[1]
    backwardCache = {}
    backwardCache["dA"+str(len(layers)-1)] = (1/m) * np.sum(((Y)/(forwardCache["A"+str(len(layers)-1)])) - ((1-Y)/(1-forwardCache["A"+str(len(layers)-1)])))
    for i in range(len(layers)-1, 0):
        backwardCache["dZ"+str(i)] = backwardCache["dA"+str(i)] * diffRelu(forwardCache["Z"+str(i)])
        backwardCache["dW"+str(i)] = (1/m) * np.dot(backwardCache["dZ"+str(i)], (forwardCache["A"+str(i-1)]).T)
        backwardCache["db"+str(i)] = (1/m) * np.sum(backwardCache["dZ"+str(i)], axis=1, keepdims=True)
        backwardCache["dA"+str(i-1)] = np.dot((forwardCache["W"+str(i)]).T, backwardCache["dZ"+str(i)])
    return backwardCache

def eff(Y_gen, Y):
    m = Y.shape[1]
    e = np.sum((1/m)(np.divide(Y - Y_gen, Y)))
    return e

def preprocess():
    # load and prepare data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train, Y_train = flatten(x_train, y_train)
    X_test, Y_test = flatten(x_test, y_test)
    return X_train, Y_train, X_test, Y_test

def model(X_train, Y_train, X_test, Y_test, layers, alpha, times):
    # setup hyperparams
    n0 = X_train.shape[0]
    m = X_train.shape[1]
    layers[0] = n0
    # initialize and execute
    costs = []
    params = initParams(layers)
    for i in range(times):
        print("iteration ", i)
        forwardCache = forwardProp(X_train, params, layers)
        backwardCache = backwardProp(X_train, Y_train, params, forwardCache, layers)
        params = updateParams(params, backwardCache, layers, alpha)
        costs.append(cost(Y_train, forwardCache["A" + str(len(layers) - 1)], m))
    # plot costs
    plt.plot(costs)
    plt.show()
    # train data result
    forwardCache = forwardProp(X_train, params, layers)
    Y_gen = forwardCache["A" + str(len(layers) - 1)]
    efficiency = eff(Y_gen, Y_train)
    print("The efficiency for the TRAIN case is: ", efficiency)
    # test data result
    forwardCache = forwardProp(X_test, params, layers)
    Y_gen_2 = forwardCache["A" + str(len(layers) - 1)]
    efficiency = eff(Y_gen_2, Y_test)
    print("The efficiency for the TEST case is: ", efficiency)

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = preprocess()
    model(X_train, Y_train, X_test, Y_test, [0, 5, 4, 3, 4, 5, 10], 0.1, 100)