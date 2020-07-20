#!/usr/local/bin/python
import deep as d
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from numba import njit

# @njit
def filter(array, filterSize): 
    filterBasic = np.ones((filterSize, filterSize))
    # filterBasic = [[1,0,0],[0,1,0],[0,0,1]]
    array = np.pad(array, int((filterSize - array.shape[0]%filterSize)/2))
    test = np.ones((int(array.shape[0]/filterSize), int(array.shape[1]/filterSize)))
    filterFull = np.outer(test, filterBasic).reshape(30,30)
    filteredArray = array * filterFull
    cache = np.zeros((filterSize, filterSize))
    arrayFinal = np.zeros((int(filteredArray.shape[0]/filterSize), int(filteredArray.shape[1]/filterSize)))
    # filter moving over the matrix
    for i in range(0, filteredArray.shape[0], filterSize):
        for j in range(0, filteredArray.shape[0], filterSize):
            cache = filteredArray[i:i+filterSize, j:j+filterSize]
            arrayFinal[int(i/filterSize)][int(j/filterSize)] = np.linalg.det(cache)
    return arrayFinal

# @njit
def pool(array, stride):
    cache = np.zeros((stride, stride))
    arrayFinal = np.zeros((int(array.shape[0]/stride), int(array.shape[1]/stride)))
    # window moving over the matrix
    for i in range(0, array.shape[0], stride):
        for j in range(0, array.shape[0], stride):
            cache = array[i:i+stride, j:j+stride]
            arrayFinal[int(i/stride)][int(j/stride)] = np.amax(cache)
    return arrayFinal

def conv(array, filterSize, stride):
    array = filter(array, filterSize)
    array = pool(array, stride)
    return array

# @njit
def allInOne(filterSize, stride):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    a = x_train.shape[1]
    f = filterSize
    s = stride
    pix = int((a + f - a%f)/(f * s))
    x_train_1 = np.zeros((x_train.shape[0], pix, pix))
    x_test_1 = np.zeros((x_test.shape[0], pix, pix))
    # for all the images in the dataset
    for i in range(x_train.shape[0]):
        x_train_1[i] = conv(x_train[i], f, s)
        if i in range(x_test.shape[0]):
            x_test_1[i] = conv(x_test[i], f, s)
    X_train, Y_train = d.flatten(x_train_1, y_train)
    X_test, Y_test = d.flatten(x_test_1, y_test)
    plt.plot()
    plt.imshow(x_train_1[0])
    plt.show()
    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = allInOne(3,2)
    # d.model(X_train, Y_train, X_test, Y_test, [0, 5, 4, 3, 4, 5, 10], 0.1, 100)