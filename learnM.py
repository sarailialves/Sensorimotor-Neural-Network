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


class SensorimotorM():
    def __init__(self, hiddenM, hiddenS, q, y, Pi, M, lr, lrp):
        movebase = len(q)
        self.M = M
        self.P = Pi
        self.lr = lr
        self.lrp = lrp

    def forward(self, q):
        self.a = np.dot(q, self.M)
        self.out = np.dot(self.a, self.P)
        return self.out

    def backward(self, q, y, out):
        self.out_error = (y - out)
        self.p_error = np.dot(self.a.T, self.out_error)

        self.out_error = np.dot(self.out_error, self.P.T)

        self.out_error = np.dot(q.T, self.out_error)

        self.P = self.P + self.lrp * self.p_error
        self.P = relu(self.P)

        self.M = self.M + self.lr * self.out_error
        self.M = relu(self.M)
        self.M = self.M / np.linalg.norm(self.M)

    def train(self, q, y):
        yy = self.forward(q)
        self.backward(q, y, yy)
