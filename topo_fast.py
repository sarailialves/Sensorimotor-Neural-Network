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
import math
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.spatial import Voronoi, voronoi_plot_2d
import warnings
import matplotlib.cbook

warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)

def centeroidnp(arr):
    length = len(arr)
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x / length, sum_y / length


def clusterS(matrixS, hiddenS, pix):
    retina = np.zeros((pix * pix))
    for i in range(pix * pix):
        index = 0
        high = 0.
        for j in range(hiddenS):
            if matrixS[i, j] > high:
                high = matrixS[i, j]
                index = j
        retina[i] = index
    retina = retina.reshape((pix, pix))
    allclusters = []
    for index in range(hiddenS):
        cluster = []
        for i in range(pix):
            for j in range(pix):
                if retina[i, j] == index:
                    cluster.append([i, j])
        if len(cluster) > 5:
            allclusters.append(centeroidnp(np.array(cluster)))
    return np.array(allclusters)


def clusterP(matrixS, matrixP, hiddenS, pix):
    retina = np.zeros((pix * pix))
    for i in range(pix * pix):
        index = 0
        high = 0.
        for j in range(hiddenS):
            if matrixS[i, j] > high:
                high = matrixS[i, j]
                index = j
        retina[i] = index
    retina = retina.reshape((pix, pix))
    allclusters = np.zeros((hiddenS, 2))
    for index in range(hiddenS):
        cluster = [[0, 0]]
        for i in range(pix):
            for j in range(pix):
                if retina[i, j] == index:
                    cluster.append([i, j])
        if len(cluster) < 10:
            cluster = [[0, 0]]
        allclusters[index] = centeroidnp(np.array(cluster))
    return allclusters


def clusterM(matrixM, hiddenM, movebase, roll):
    draw = np.random.rand(movebase, 3)
    for k in range(movebase):
        movement = list(roll[k])
        x = movement[0]
        y = movement[1]
        index = 0
        high = 0
        for j in range(hiddenM):
            if matrixM[k, j] > high:
                high = matrixM[k, j]
                index = j
        draw[k] = x, y, index
    allclusters = []
    for index in range(hiddenM):
        cluster = []
        for k in range(movebase):
            if draw[k, 2] == index:
                cluster.append([draw[k, 0], draw[k, 1]])
        if len(cluster) > 2:
            allclusters.append(centeroidnp(np.array(cluster)))
    return np.array(allclusters)


# Auxiliary function from: [https://stackoverflow.com/a/20678647/1595060]
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all([v >= 0 for v in vertices]):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)


def topoP(S, P, number, pix, hiddenS):
    mpl.style.use('seaborn-white')
    fig = plt.figure(figsize=(6, 6))

    clusters = clusterP(S, P, hiddenS, pix)
    vor = Voronoi(clusters)
    points = np.array(clusters)
    movements = np.random.rand(hiddenS, 2)
    for i in range(hiddenS):
        fro = i
        high = 0.0
        to = i
        for j in range(hiddenS):
            if P[i, j] > high:
                high = P[i, j]
                to = j
        movements[i] = np.array([fro, to])
    for i in range(hiddenS):
        fro = int(movements[i][0])
        tom = int(movements[i][1])
        initial = clusters[fro]
        end = clusters[tom]
        y1 = end[0] - initial[0]
        y2 = end[1] - initial[1]
        if initial[0] != 0 and initial[1] != 0 and end[0] != 0 and end[1] != 0 and fro != tom:
            factor = 1
            factor = sqrt(1 / (math.pow(end[0] - initial[0], 2) + math.pow(end[1] - initial[1], 2)))
            plt.axes().arrow(initial[0], initial[1], factor * y1, factor * y2, width=0.1, head_width=0.6, color='black')
    plt.axis('off')
    plt.xlim(0, pix)
    plt.xticks(())
    plt.ylim(0, pix)
    plt.yticks(())
    pylab.savefig('pfield/p' + str(number) + '.png', bbox_inches='tight')
    plt.close()


def topoS(matrixS, ind, pix, hiddenS):
    retina = np.random.rand(pix * pix, 4)
    colors = mpl.cm.rainbow(np.linspace(0, 1, hiddenS))

    for i in range(pix * pix):
        index = 0
        high = 0
        for j in range(hiddenS):
            if matrixS[i, j] > high:
                high = matrixS[i, j]
                index = j
        retina[i] = colors[index]

    retina = retina.reshape((pix, pix, 4))

    mpl.style.use('seaborn-white')
    fig = plt.figure(figsize=(7, 7))
    clusters = clusterS(matrixS, hiddenS, pix)
    vor = Voronoi(clusters)

    regions, vertices = voronoi_finite_polygons_2d(vor, 35)
    k = 0
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), fill=None, alpha=0.7, edgecolor='gray', linewidth='1.5')

        k += 1
    for k in range(len(clusters)):
        plt.plot(clusters[k][0], clusters[k][1], 'o', color='black', markersize=7)

    plt.grid(False)
    plt.xlim(-1, pix)
    plt.xticks(())
    plt.ylim(-1, pix)
    plt.yticks(())
    plt.axis('off')
    pylab.savefig('sfield/s' + str(ind) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def topoM1(matrixM, mov, roll, hiddenM, movebase):
    mpl.style.use('seaborn-white')
    fig = plt.figure(figsize=(7, 7))

    clusters = clusterM(matrixM, hiddenM, movebase, roll)
    vor = Voronoi(clusters)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=200)
    k = 0
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), fill=None, alpha=0.7, edgecolor='gray', linewidth='1.5')
        k += 1

    for k in range(len(clusters)):
        plt.plot(clusters[k][0], clusters[k][1], 'o', color='black', markersize=5)

    plt.xticks([])
    plt.yticks([])
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.grid(False)
    plt.axis('off')
    pylab.savefig('mfield/m' + str(mov) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


def topoM2(matrixM, mov, roll, hiddenM, movebase):
    mpl.style.use('seaborn-white')
    fig = plt.figure(figsize=(7, 7))

    clusters = clusterM(matrixM, hiddenM, movebase, roll)
    vor = Voronoi(clusters)
    regions, vertices = voronoi_finite_polygons_2d(vor, radius=200)
    k = 0
    for region in regions:
        polygon = vertices[region]
        plt.fill(*zip(*polygon), fill=None, alpha=0.7, edgecolor='gray', linewidth='1.5')
        k += 1

    for k in range(len(clusters)):
        plt.plot(clusters[k][0], clusters[k][1], 'o', color='black', markersize=5)

    plt.xticks([])
    plt.yticks([])
    plt.xlim(0, 6)
    plt.ylim(0, 6)
    plt.grid(False)
    plt.axis('off')
    pylab.savefig('mfield/m' + str(mov) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()
