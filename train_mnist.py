import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('C:/Users/Ojas/.cache/kagglehub/datasets/crawford/emnist/versions/3/emnist-balanced-train.csv', header=None)


data = np.array(data)
m, n = data.shape
np.random.shuffle(data)


data_dev = data[0:2000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n] 

data_train = data[2000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def initparams():
    W1 = np.random.rand(47, 784) * 0.01
    b1 = np.zeros((47, 1))
    W2 = np.random.rand(47, 47) * 0.01
    b2 = np.zeros((47, 1))

    return W1, b1, W2, b2


def prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = Softmax(Z2)
    

    return Z1, A1, Z2, A2

def ReLU(Z):
    return np.maximum(0, Z)

def Softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)


