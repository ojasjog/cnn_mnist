import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('emnist-balanced-train.csv', header=None)


data = np.array(data)
m, n = data.shape

np.random.shuffle(data)


data_dev = data[0:1000].T

Y_dev = data_dev[0]
X_dev = data_dev[1:n] / 255.0

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n] / 255.0



emnist_mapping = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'
]


def visualize_predictions(W1, b1, W2, b2, X_data, Y_data, num_samples=10):
    _, _, _, A2 = prop(W1, b1, W2, b2, X_data)
    
    raw_predictions = np.argmax(A2, axis=0)

    random_indices = np.random.choice(Y_data.size, num_samples, replace=False)
    
    print("\n" + "="*40)
    print(f"      VISUALIZING {num_samples} RANDOM PREDICTIONS")
    print("="*40)
    
    for idx in random_indices:
        pred_index = raw_predictions[idx]
        true_index = Y_data[idx]
    
        predicted_char = emnist_mapping[pred_index]
        true_char = emnist_mapping[true_index]
        
        status = "Predicted" if predicted_char == true_char else f"Not Predicted (Should be {true_char})"
        
        print(f"Sample #{idx:4d} | Predicted: [ {predicted_char} ] | True: [ {true_char} ] | {status}")


def initparams():
    W1 = np.random.randn(64, 784) * np.sqrt(2.0 / 784)
    b1 = np.zeros((64, 1))

    W2 = np.random.randn(47, 64) * np.sqrt(1.0 / 64)
    b2 = np.zeros((47, 1))

    return W1, b1, W2, b2


def prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = Softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(Y):
    Y_onehot = np.zeros((Y.size, 47))
    Y_onehot[np.arange(Y.size), Y] = 1
    return Y_onehot.T
def backprop(W1, b1, W2, b2, Z1, A1, Z2, A2, X, Y):
    m = Y.size
    Y_onehot = one_hot(Y) 
    dZ2 = A2 - Y_onehot
    dW2 = (dZ2.dot(A1.T)) / m
    db2 = np.sum(dZ2, 1, keepdims=True) / m

    dA1 = W2.T.dot(dZ2)
    dZ1 = dA1 * deriv_Relu(Z1)
    dW1 = (dZ1.dot(X.T)) / m
    db1 = np.sum(dZ1, 1, keepdims=True) / m

    return dW1, db1, dW2, db2

def updateparams(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def Softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / np.sum(expZ, axis=0, keepdims=True)

def deriv_Relu(Z):
    return (Z > 0)

def get_predictions(A2):
    return np.argmax(A2, axis=0)


def gradient_descent(X, Y, learning_rate, iterations):
    W1, b1, W2, b2 = initparams()
    for i in range(iterations):
        Z1, A1, Z2, A2 = prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backprop(W1, b1, W2, b2, Z1, A1, Z2, A2, X, Y)
        W1, b1, W2, b2 = updateparams(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)

        print(f"Iteration {i}")
        accuracy = np.mean(np.argmax(A2, axis=0) == Y)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Predictions: {get_predictions(A2)[:10]}")

    return W1, b1, W2, b2

learning_rate = 0.01
iterations = 1000
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, learning_rate, iterations)
Z1_dev, A1_dev, Z2_dev, A2_dev = prop(W1, b1, W2, b2, X_dev)
accuracy_dev = np.mean(np.argmax(A2_dev, axis=0) == Y_dev)

print(f"Development Set Accuracy: {accuracy_dev:.4f}")

visualize_predictions(W1, b1, W2, b2, X_dev, Y_dev, num_samples=15)

