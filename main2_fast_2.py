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
from topo_fast import topoM2
from tqdm import tqdm
import math
import itertools
import math
from scipy.spatial import Voronoi, voronoi_plot_2d
import warnings
import matplotlib.cbook

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

f = open('pickle/weights_fast.pckl', 'rb')
S, M, P, Pi, hiddenS, hiddenM = pickle.load(f)
f.close()

# Inputs
lrm = input("[Enter] Learning rate for M: ")
lrm = float(lrm)
lrp = input("[Enter] Learning rate for P: ")
lrp = float(lrp)
epoch = input("[Enter] Number of iterations: ")
epoch = int(epoch)

# Training

inputdata = [[3, 4, 5, 6, 0, 2, 1],
             [3, 4, 5, 6, 2, 1, 0]]
roll = list(itertools.product(*inputdata))


def forwardSN(i, qi):
    maxi = len(i)
    novovec = np.zeros((maxi, pix * pix))
    for k in range(maxi):
        a = np.dot(i[k], S)
        qnew = int(qi[k])
        b = np.dot(q[qnew], M)
        c = np.dot(b, Pi)
        c = c.reshape((hiddenS, hiddenS))
        d = np.dot(a, c)
        novovec[k] = np.dot(d, S.T)
    return novovec


P = P.reshape((movebase, hiddenS * hiddenS))

q = np.zeros((movebase, movebase))
for k in range(movebase):
    q[k][k] = 1

y = P

topoM2(M, 0, roll, hiddenM, movebase)
mov = 1

SN = SensorimotorM(hiddenM, hiddenS, q, y, Pi, M, lrm, lrp)
j = 0
while j < epoch + 1:
    SN.train(q, y)
    M = SN.M
    Pi = SN.P
    if j % (epoch / 10) == 0:
        topoM2(SN.M, mov, roll, hiddenM, movebase)
        mov += 1
    j += 1

print("::: Learning process finished")
rmse = np.sqrt(np.mean((plot_y_train - forwardSN(plot_i_train, plot_q_train)) ** 2))
print("::: RMSE:", rmse)

# Saving the results
f = open('pickle/weights2_fast.pckl', 'wb')
pickle.dump([M, Pi], f)
f.close()
