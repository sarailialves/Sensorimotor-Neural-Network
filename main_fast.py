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
from learnS import SensorimotorS
from learnP import SensorimotorP
from learnM import SensorimotorM
from topo_fast import topoS, topoP
from tqdm import tqdm
import math

# Importing previously generated data
f = open('pickle/data1.pckl', 'rb')
all_i_train, all_y_train, all_i_test, all_y_test = pickle.load(f)
f.close()

f = open('pickle/data2.pckl', 'rb')
plot_i_train, plot_y_train, plot_q_train, plot_x_train, plot_i_test, plot_y_test, plot_q_test, plot_x_test = pickle.load(
    f)
f.close()

f = open('pickle/common1.pckl', 'rb')
movebase, pix = pickle.load(f)
f.close()

f = open('pickle/save.pckl', 'rb')
S, Pi, M, movebase, hiddenS, hiddenM, pixel = pickle.load(f)
f.close()

# Inputs
lrs = input("[Enter] Learning rate for S: ")
lrs = float(lrs)
lrp = input("[Enter] Learning rate for predictors: ")
lrp = float(lrp)
maxim_iter = input("[Enter] Number of iterations: ")
maxim_iter = int(maxim_iter)
epoch = maxim_iter // 10

# Training
P = np.zeros((movebase, hiddenS, hiddenS))
Pi = Pi.reshape((hiddenM, hiddenS * hiddenS))
q = np.zeros((movebase, movebase))
for k in range(movebase):
    q[k, k] = 1
    aux = np.dot(q[k], M)
    aux = np.dot(aux, Pi)
    P[k] = aux.reshape((hiddenS, hiddenS))


def relu(x):
    return x * (x > 0)


topoS(S, 0, pix, hiddenS)
retina = 1
iterations = list(range(maxim_iter))


def forwardSN(i, q):
    maxi = len(i)
    novovec = np.zeros((maxi, pix * pix))
    for k in range(maxi):
        a = np.dot(i[k], S)
        index = int(q[k])
        b = np.dot(a, P[index])
        novovec[k] = np.dot(b, S.T)
    return novovec


for iterate in tqdm(iterations):

    # Training S
    vector = 1
    for mov in range(movebase):
        i_train = all_i_train[mov]
        y_train = all_y_train[mov]
        i_test = all_i_test[mov]
        y_test = all_y_test[mov]

        SN = SensorimotorP(i_train, y_train, i_test, y_test, pix, hiddenS, S, P[mov], lrp)

        j = 0

        while j < epoch + 1:
            SN.train(i_train, y_train)
            j += 1
        P[mov] = SN.P

    # Training P

    for mov in range(movebase):

        i_train = all_i_train[mov]
        y_train = all_y_train[mov]
        i_test = all_i_test[mov]
        y_test = all_y_test[mov]

        SN = SensorimotorS(i_train, y_train, i_test, y_test, pix, hiddenS, S, P[mov], lrs)
        j = 0
        while j < epoch + 1:
            SN.train(i_train, y_train)
            j += 1
        S = SN.S
    if iterate % math.floor(maxim_iter / 10) == 0:
        topoS(SN.S, retina, pix, hiddenS)
        retina += 1

    if iterate % math.floor(maxim_iter / 10) == 0:
        vector = 1
        for k in range(movebase):
            topoP(S, P[k], vector, pix, hiddenS)
            vector += 1

print("::: Learning process finished")
rmse = (np.sqrt(np.mean(np.square(plot_y_train - forwardSN(plot_i_train, plot_q_train)))))
print("::: RMSE: ", rmse)

vector = 1
for k in range(movebase):
    topoP(S, P[k], vector, pix, hiddenS)
    vector += 1
topoS(S, retina, pix, hiddenS)

# Saving the results
f = open('pickle/weights_fast.pckl', 'wb')
pickle.dump([S, M, P, Pi, hiddenS, hiddenM], f)
f.close()
