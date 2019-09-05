import numpy as np
import pylab
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.cm as cm
import pickle
import time
import matplotlib as mpl
from sklearn.decomposition import NMF
import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
import math

# Importing previously generated data
f = open('pickle/common1.pckl', 'rb')
movebase, pix = pickle.load(f)
f.close()
f = open('pickle/B.pckl', 'rb')
B = pickle.load(f)
f.close()

# Inputs
hiddenS = input("[Enter] Number of visual hidden nodes: ")
hiddenS = int(hiddenS)
hiddenM = input("[Enter] Number of motor hidden nodes: ")
hiddenM = int(hiddenM)


# Defining auxiliary function
def NNLS(M, U):
    N, p1 = M.shape
    q, p2 = U.shape
    X = np.zeros((N, q), dtype=np.float32)
    MtM = np.dot(U, U.T)
    for n1 in range(N):
        X[n1] = scipy.optimize.nnls(MtM, np.dot(U, M[n1]))[0]
    return X


# First factorization
model = NMF(n_components=hiddenM, init='random', random_state=0)
W = model.fit_transform(B.T)
H = model.components_
Mi = H.T
W = W.T

# Transformation
W = np.vsplit(W, hiddenM)
novoW = W[0].reshape((pix * pix, pix * pix))
for i in range(1, len(W)):
    aux = W[i].reshape((pix * pix, pix * pix))
    novoW = np.concatenate((novoW, aux), axis=1)
W = novoW

# Second Factorization
model = NMF(n_components=hiddenS, init='random', random_state=0)
C = model.fit_transform(W)
B = model.components_
Si = C

# Another transformation
B = np.hsplit(B, hiddenM)
novoB = B[0]
for i in range(1, len(B)):
    novoB = np.concatenate((novoB, B[i]), axis=0)
B = novoB

# Aplying NNLS to find P
X = NNLS(B, Si.T)
Pi = np.zeros((hiddenM, hiddenS * hiddenS))
novoX = np.hsplit(X, hiddenM)
for i in range(hiddenM):
    aux = novoX[i].reshape((hiddenS * hiddenS))

print("::: Factorization complete.")

# Saving the results
f = open('pickle/save.pckl', 'wb')
pickle.dump([Si, Pi, Mi, movebase, hiddenS, hiddenM, pix], f)
f.close()
