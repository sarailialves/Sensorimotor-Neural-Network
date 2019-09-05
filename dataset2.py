import numpy as np
from PIL import Image
import matplotlib as mpl
import pickle
import random
import itertools
import math
from scipy import ndimage
import os

# Inputs
pix = input("[Enter] Retina's width/height (must be an even number): ")
pix = int(pix)
train_batch = input("[Enter] Size of training batch: ")
train_batch = int(train_batch)
test_batch = input("[Enter] Size of testing batch: ")
test_batch = int(test_batch)
batch = train_batch + test_batch
my_path = os.path.abspath(os.path.dirname(__file__))
file_name = input('[Enter] Relative path of image to be used (Example: image.jpg): ')
directory = os.path.join(my_path, file_name)
pixi = pix // 2
batch = train_batch + test_batch

# Generating dataset
inputdata = [[0, 30, 60, 90, -90, -30, -60],
             [0, 1, 2, 3, -1, -2, -3]]
roll = list(itertools.product(*inputdata))
movebase = len(roll)


def rotation(img, angle, pivot):
    padX = [img.size[1] - pivot[0], pivot[0]]
    padY = [img.size[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0]: -padY[1], padX[0]: -padX[1]]


def image_to_array(image):
    arrayi = list(image.getdata())
    arrayi = np.array(arrayi)
    return arrayi


def gen(train_batch, test_batch, directory, pix):
    img = Image.open(directory).convert('L')
    size = img.size[0]

    label = np.zeros((movebase, movebase))
    for i in range(movebase):
        label[i][i] = 1

    i_train = np.zeros((movebase, train_batch, pix * pix))
    y_train = np.zeros((movebase, train_batch, pix * pix))
    i_test = np.zeros((movebase, test_batch, pix * pix))
    y_test = np.zeros((movebase, test_batch, pix * pix))

    plot_i_train = np.zeros((train_batch * movebase, pix * pix))
    plot_y_train = np.zeros((train_batch * movebase, pix * pix))
    plot_q_train = np.zeros((train_batch * movebase))
    plot_x_train = np.zeros((train_batch * movebase, pix * pix + movebase + 1))

    plot_i_test = np.zeros((test_batch * movebase, pix * pix))
    plot_y_test = np.zeros((test_batch * movebase, pix * pix))
    plot_q_test = np.zeros((test_batch * movebase))
    plot_x_test = np.zeros((test_batch * movebase, pix * pix + movebase + 1))

    count_train = 0
    count_test = 0
    count = 0
    count2 = 0

    aux = math.ceil(math.sqrt(math.ceil(train_batch + test_batch)))
    aux2 = math.floor((size - pix * 5) / aux)

    ratio = train_batch / (train_batch + test_batch)
    for j in range(2 * pix, size - 2 * pix, aux2):
        for i in range(2 * pix, size - 2 * pix, aux2):
            area = (i - pixi, j - pixi, i + pixi, j + pixi)
            cropped_img = img.crop(area)
            initial_img = image_to_array(cropped_img)

            randn = np.random.rand(1)

            if count_train < train_batch and (randn < ratio and count_train < train_batch) or (
                    count_train < train_batch and count_test > test_batch):
                for index in range(movebase):
                    mov = list(roll[index])
                    rotimg = rotation(img, mov[0], [i, j])
                    rotimg = Image.fromarray(rotimg)
                    area2 = (i - mov[1] - pixi, j - mov[1] - pixi, i + mov[1] + pixi, j + mov[1] + pixi)
                    rotimg = rotimg.crop(area2)
                    rotimg = rotimg.resize((pix, pix), Image.BILINEAR)
                    transf_img = image_to_array(rotimg)
                    plot_q_train[count] = index
                    plot_i_train[count] = initial_img / 255
                    plot_y_train[count] = transf_img / 255
                    plot_x_train[count] = np.append(np.append(initial_img / 255, label[index]), 1)
                    count += 1
                    i_train[index, count_train] = initial_img / 255
                    y_train[index, count_train] = transf_img / 255
                count_train += 1

            if count_test < test_batch and (randn > ratio and count_test < test_batch) or (
                    count_test < test_batch and count_train > train_batch):
                for index in range(movebase):
                    mov = list(roll[index])
                    rotimg = rotation(img, mov[0], [i, j])
                    rotimg = Image.fromarray(rotimg)
                    area2 = (i - mov[1] - pixi, j - mov[1] - pixi, i + mov[1] + pixi, j + mov[1] + pixi)
                    rotimg = rotimg.crop(area2)
                    rotimg = rotimg.resize((pix, pix), Image.BILINEAR)
                    transf_img = image_to_array(rotimg)
                    plot_q_test[count2] = index
                    plot_i_test[count2] = initial_img / 255
                    plot_y_test[count2] = transf_img / 255
                    plot_x_test[count2] = np.append(np.append(initial_img / 255, label[index]), 1)
                    count2 += 1
                    i_test[index, count_test] = initial_img / 255
                    y_test[index, count_test] = transf_img / 255
                count_test += 1

    return i_train, y_train, i_test, y_test, plot_i_train, plot_y_train, plot_q_train, plot_x_train, plot_i_test, plot_y_test, plot_q_test, plot_x_test


i_train, y_train, i_test, y_test, plot_i_train, plot_y_train, plot_q_train, plot_x_train, plot_i_test, plot_y_test, plot_q_test, plot_x_test = gen(
    train_batch, test_batch, directory, pix)

print("::: Dataset generated.")

# Saving the results
f = open('pickle/data1.pckl', 'wb')
pickle.dump([i_train, y_train, i_test, y_test], f)
f.close()

f = open('pickle/common1.pckl', 'wb')
pickle.dump([movebase, pix], f)
f.close()

f = open('pickle/data2.pckl', 'wb')
pickle.dump(
    [plot_i_train, plot_y_train, plot_q_train, plot_x_train, plot_i_test, plot_y_test, plot_q_test, plot_x_test], f)
f.close()
