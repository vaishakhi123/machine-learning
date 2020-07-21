#!/usr/local/bin/python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist, fashion_mnist
import pickle
import os

"""
Three functions, relu, sigmoid and tanh. And their differentiations.
"""

def relu(z):
    return z * (z > 0)

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def diffSigmoid(z):
    return sigmoid(z)*(1-sigmoid(z))

def diffRelu(z):
    return relu(z)/z

"""
initParams: Initialise parameters with random numbers for W and zeros for b
    params: Contains W and b
flatten: Flatten it to 2d, and transposing to get shape (x**2, m)
preprocess: Load data and flatten them for final input
"""

def initParams(layers): 
    params = {}
    for i in range(1, len(layers)):
        params["W" + str(i)] = (np.random.rand(layers[i], layers[i-1])) * 0.01 # Ws are independent of m
        params["b" + str(i)] = np.zeros((layers[i], 1)) # b gets broadcasted to remove dependence on m for later use
    return params

def flatten(x_train, y_train, options):
    # mnist dataset comes in the shape (m, x, y), without color
    # with color stuff can be just flattened with a * 3 at the end
    x = (x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2]))/255
    y = (y_train.reshape(y_train.shape[0], options))/100
    return x.T, y.T+0.01

def preprocess(options):
    # load and prepare data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    X_train, Y_train = flatten(x_train, y_train, options)
    X_test, Y_test = flatten(x_test, y_test, options)
    return X_train, Y_train, X_test, Y_test

"""
forwardProp: Forward propagation function
    forwardCache: Contains A and Z
backwardProp: calculating the changes in all parameters to finally bring cost down
    backwardCache: Contains dA, dZ, dW and db
updateParams: update W and b with hyperparameter alpha
"""

def forwardProp(X, params, layers):
    forwardCache = {}
    forwardCache["A0"] = X + 0.1
    for i in range(1, len(layers)):
        forwardCache["Z" + str(i)] = np.dot(params["W" + str(i)], forwardCache["A" + str(i-1)]) + params["b" + str(i)]
        forwardCache["A" + str(i)] = sigmoid(forwardCache["Z" + str(i)]) # using relu for all activations
    return forwardCache

def backwardProp(X, Y, params, forwardCache, layers):
    m = X.shape[1]
    backwardCache = {}
    backwardCache["dA"+str(len(layers)-1)] = (-1/m) * np.sum(((Y)/(forwardCache["A"+str(len(layers)-1)])) - ((1-Y)/(1-forwardCache["A"+str(len(layers)-1)])))
    for i in reversed(range(1, len(layers))):
        backwardCache["dZ"+str(i)] = backwardCache["dA"+str(i)] * diffSigmoid(forwardCache["Z"+str(i)])
        backwardCache["dW"+str(i)] = (1/m) * np.dot(backwardCache["dZ"+str(i)], (forwardCache["A"+str(i-1)]).T)
        backwardCache["db"+str(i)] = (1/m) * np.sum(backwardCache["dZ"+str(i)], axis=1, keepdims=True)
        backwardCache["dA"+str(i-1)] = np.dot((params["W"+str(i)]).T, backwardCache["dZ"+str(i)])
    return backwardCache

def updateParams(params, backwardCache, layers, alpha):
    for i in range(1, len(layers)):
        params["W" + str(i)] -= alpha * backwardCache["dW" + str(i)]
        params["b" + str(i)] -= alpha * backwardCache["db" + str(i)]
    return params

"""
cost: Calculate the logarithmic cost value for a set of W and b
eff: Compare and calculate the efficiency for the set
"""

def cost(Y, A, m):
    # J = np.sum(np.dot((A-Y).T, (A-Y)))
    J = (-1/m) * np.sum(np.dot(Y, np.log(A).T) + np.dot(1 - Y, np.log(1 - A).T))  
    return J

def eff(Y_gen, Y, m):
    
    p = (1/m)*(np.sum(np.abs(Y_gen - Y)))
    return (1-p)*100

"""
model: Aggregate of all the functions at a time, and running the optimizations a number of time to get the best possible value
    Hyperparameters like the layers, alpha, number of times needed to minimize cost are to be set here. Even to be used inside other functions. 
"""
def model(X_train, Y_train, X_test, Y_test, layers, alpha, times):
    # setup hyperparams
    n0 = X_train.shape[0]
    m = X_train.shape[1]
    mb = X_test.shape[1]
    layers[0] = n0
    # initialize and execute
    costs = []
    params = initParams(layers)
    for i in range(times):
        print(".")
        forwardCache = forwardProp(X_train, params, layers)
        backwardCache = backwardProp(X_train, Y_train, params, forwardCache, layers)
        params = updateParams(params, backwardCache, layers, alpha)
        # print (cost(Y_train, forwardCache["A" + str(len(layers) - 1)], m))
        costs.append(cost(Y_train, forwardCache["A" + str(len(layers) - 1)], m))
    with open("feedforward.pkl", "wb") as p:
        pickle.dump(params, p)
    # plot costs
    plt.plot(costs)
    plt.show()
    # train data result
    forwardCache = forwardProp(X_train, params, layers)
    Y_gen = forwardCache["A" + str(len(layers) - 1)]
    efficiency = eff(Y_gen, Y_train, m)
    print("The efficiency while TRAINING is: ", efficiency)
    # test data result
    forwardCache = forwardProp(X_test, params, layers)
    Y_gen_2 = forwardCache["A" + str(len(layers) - 1)]
    efficiency = eff(Y_gen_2, Y_test, mb)
    print("The efficiency while TESTING case is: ", efficiency)

def modelExists(params, X_train, Y_train, X_test, Y_test, layers):
    # setup hyperparams
    n0 = X_train.shape[0]
    m = X_train.shape[1]
    mb = X_test.shape[1]
    layers[0] = n0
    # train data result
    forwardCache = forwardProp(X_train, params, layers)
    Y_gen = forwardCache["A" + str(len(layers)-1)]
    efficiency = eff(Y_gen, Y_train, m)
    print("The efficiency while TRAINING is: ", efficiency)
    # test data result
    forwardCache = forwardProp(X_test, params, layers)
    Y_gen_2 = forwardCache["A" + str(len(layers)-1)]
    efficiency = eff(Y_gen_2, Y_test, mb)
    print("The efficiency while TESTING case is: ", efficiency)

if __name__ == "__main__":
    opt = 1
    X_train, Y_train, X_test, Y_test = preprocess(opt)
    if os.path.exists("feedforward.pkl"):
        with open("feedforward.pkl", "rb") as p:
            params = pickle.load(p)
        modelExists(params, X_train, Y_train, X_test, Y_test, [0, 5, 4, 3, 4, 5, opt])
    else:
        model(X_train, Y_train, X_test, Y_test, [0, 5, 4, 3, 4, 5, opt], 1, 10)