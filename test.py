import numpy as np
from layers import *
import torch
from torch import nn
from sklearn.preprocessing import OneHotEncoder
from sklearn import datasets
from sklearn.metrics import accuracy_score

data = np.load("mnist.npy")
y = data[:, 784:785]
y = OneHotEncoder().fit_transform(y).toarray()

x = data[:, 0:784] / 256
x = x.reshape((70000, 1, 28, 28))
X_train = x[0:500]
y_train = y[0:500]
X_test = x[1000:1050]
y_test = y[1000:1050]
n = len(X_train)
layers = [Conv2D(1, 5, 2, 2, 1), Relu(), MaxPooling(2, 2), Flatten(), Dense(845, 10), SoftMax()]
max_ite = 100
alpha = 0.0001

for i in range(max_ite):
    print(f'iteration {i}')
    train_X = X_train
    for layer in layers:
        #         print(layer)
        #         print(train_X.shape)
        train_X = layer.forward(train_X)
    # print(train_X)
    E = (train_X - y_train) / n * alpha
    for k in range(len(layers) - 1, -1, -1):
        E = layers[k].backward(E)
    # print(E)
for layer in layers:
    X_test = layer.forward(X_test)
y_pred = toPredict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
