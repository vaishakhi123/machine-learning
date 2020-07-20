#!/usr/local/bin/python
import feedforward as feed
import numpy as np
from tensorflow.keras.datasets import mnist
# import matplotlib.pyplot as plt

"""
filter: does a very simple job. takes a square matrix, makes it big to the size of the actual image matrix, and multiplies elementwise.
    filterBasic: is the small basic filter matrix which gets blown up
    cache: the window which is the size of filterBasic, gets around the image matrix and takes the determinant of whatever is inside
    arrayFinal: collects the determinants from cache, eventually making a smaller matrix than the original image matrix
"""

def filter(array, filterSize): 
    filterBasic = np.ones((filterSize, filterSize))
    # filterBasic = [[1,0,0],[0,1,0],[0,0,1]]
    makeBig = np.ones((int(array.shape[0]/filterSize), int(array.shape[1]/filterSize)))
    filterFull = np.outer(makeBig, filterBasic).reshape(array.shape[0],array.shape[1])
    filteredArray = array * filterFull
    cache = np.zeros((filterSize, filterSize))
    arrayFinal = np.zeros((int(filteredArray.shape[0]/filterSize), int(filteredArray.shape[1]/filterSize)))
    # filter moving over the matrix
    for i in range(0, filteredArray.shape[0], filterSize):
        for j in range(0, filteredArray.shape[1], filterSize):
            cache = filteredArray[i:i+filterSize, j:j+filterSize]
            arrayFinal[int(i/filterSize)][int(j/filterSize)] = np.linalg.det(cache)
    return arrayFinal

"""
pool: can be of different types, max, min, average. Moves the window across the filtered matrix and takes whatever is to be taken
    cache: the actual window
    arrayFinal: after pooling matrix
"""

def pool(array, stride):
    cache = np.zeros((stride, stride))
    arrayFinal = np.zeros((int(array.shape[0]/stride), int(array.shape[1]/stride)))
    # window moving over the matrix
    for i in range(0, array.shape[0], stride):
        for j in range(0, array.shape[1], stride):
            cache = array[i:i+stride, j:j+stride]
            arrayFinal[int(i/stride)][int(j/stride)] = np.amax(cache)
    return arrayFinal

"""
conv: just puts the two things together for aesthetic purposes
    filterSize: size of the filter
    stride: thee size of thee window for the pooling process
"""

def conv(array, filterSize, stride):
    array = filter(array, filterSize)
    array = pool(array, stride)
    return array

"""
allInOne: the main function, if you might. takes initial data, converts the 3d matrices into 2d by stacking them on top of each other and passes them over to conv
    x_train, y_train and x_test, y_test: the pulled data from API. The ones with _1 are after conv, to be pushed to flattened
    pix: the final pixel count after filtering and pooling process
    X_train, Y_train and X_test, Y_test: all ready and ready to be put into feed forward network
"""

def allInOne(filterSize, stride):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    ma = x_train.shape[0]
    mb = x_test.shape[0]
    a = x_train.shape[1]
    b = x_test.shape[1]
    f = filterSize # 3
    s = stride # 2
    pix = int((a + f - a%f)/(f * s)) # 5
    pad = int((f - a%f))
    x_train = np.reshape(np.pad(x_train, ((0,0), (0, pad), (0, pad))), (ma*(a + pad), (a + pad)))
    x_test = np.reshape(np.pad(x_test, ((0,0), (0, pad), (0, pad))), (mb*(b + pad), (b + pad)))
    # for all the images in the dataset
    x_train_1 = np.reshape(conv(x_train, f, s), (ma, pix, pix))
    x_test_1 = np.reshape(conv(x_test, f, s), (mb, pix, pix))
    X_train, Y_train = feed.flatten(x_train_1, y_train)
    X_test, Y_test = feed.flatten(x_test_1, y_test)
    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = allInOne(3,2)
    feed.model(X_train, Y_train, X_test, Y_test, [0, 5, 4, 3, 4, 5, 10], 0.001, 1000)