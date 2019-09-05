import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.cm as cm
import pickle
from sklearn import preprocessing
import time

# Inputs
lr = input("[Enter] Learning rate: ")
lr = float(lr)
epoch = input("[Enter] Number of iterations: ")
epoch = int(epoch)
step = epoch // 10

# Importing previously generated data
f = open('pickle/common1.pckl', 'rb')
movebase, pix = pickle.load(f)
f.close()

f = open('pickle/data1.pckl', 'rb')
all_i_train, all_y_train, all_i_test, all_y_test = pickle.load(f)
f.close()


def relu(x):
    return x * (x > 0)


class Multilayer(object):
    def __init__(self, x, y, x_test, y_test):
        self.W = np.random.rand(pix * pix, pix * pix)
        self.W = relu(self.W)

    def forward(self, x):
        self.out = np.dot(x, self.W)
        return self.out

    def backward(self, x, y, out):
        self.out_error = y - out
        self.a = np.dot(x.T, self.out_error)
        self.W = self.W + lr * self.a

        self.W = relu(self.W)

    def train(self, x, y):
        yy = self.forward(x)
        self.backward(x, y, yy)

    def predict(self, x_test, y_test):
        predicted = self.forward(x_test)
        rsme = np.mean(np.square(y_test - predicted))
        return predicted, rsme


B = np.random.rand(movebase, pix * pix * pix * pix)

for i in range(movebase):
    itrain1 = all_i_train[i]
    ytrain1 = all_y_train[i]
    itest1 = all_i_test[i]
    ytest1 = all_y_test[i]

    MN = Multilayer(itrain1, ytrain1, itest1, ytest1)
    for j in range(epoch + 1):
        MN.train(itrain1, ytrain1)
    rmse = np.mean(np.square(ytrain1 - MN.forward(itrain1)))
    print("::: Network ", i + 1, "/", movebase, " ::: RMSE: ", rmse)
    B[i] = MN.W.reshape(pix * pix * pix * pix)

print("::: Learning process finished")

# Saving the results
f = open('pickle/B.pckl', 'wb')
pickle.dump(B, f)
f.close()
