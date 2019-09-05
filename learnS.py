import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.cm as cm
import pickle
import matplotlib as mpl
import time
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt


def relu(x):
    return x * (x > 0)


class SensorimotorS():
    def __init__(self, i, y, i_test, y_test, pix, hiddenS, S, P, lr):
        self.S = S
        self.P = P
        self.lr = lr

    def forward(self, i):
        self.a = np.dot(i, self.S)
        self.b = np.dot(self.a, self.P)
        self.out = np.dot(self.b, self.S.T)
        return self.out

    def backward(self, i, y, out):
        self.out_error = (y - out)
        self.s_error2 = np.dot(self.b.T, self.out_error)
        self.out_error = np.dot(self.out_error, self.S)
        self.out_error = np.dot(self.out_error, self.P.T)
        self.s_error = np.dot(i.T, self.out_error)

        self.S = self.S + self.lr * (self.s_error + self.s_error2.T) / 2
        self.S = self.S / np.linalg.norm(self.S)
        self.S = relu(self.S)

    def train(self, i, y):
        yy = self.forward(i)
        self.backward(i, y, yy)

    def predict(self, i_test, y_test):
        print("Testing Sensorimotor.... ")
        predicted = self.forward(i_test)
        mss2 = sqrt(mean_squared_error(y_test, predicted))
        print("MSS: ", mss2)
        return predicted, mss2
