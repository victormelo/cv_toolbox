# encoding: utf-8

from core.util import get_histogram
import matplotlib.pyplot as plt
import numpy as np
import Image
import os
from skimage.transform import resize
import scipy

def pca(img):
    h, w = img.shape
    mu = np.mean(img, axis=0)
    dif = img-mu

    cm = np.dot(dif.T, dif) / (h-1)
    [U, S, V] = np.linalg.svd(cm)
    return dif, U, S

def main():

    PATH = '../images/cropped_faces/'
    img = list()
    shape = (32, 32)
    cont = 50

    for path in os.listdir(PATH):
        i = np.array(Image.open(PATH+path).convert('L').resize(shape, Image.ANTIALIAS))
        img.append(i.flatten())
        cont -= 1

        if(cont<=0):
            break

    img = np.array(img)

    dif, U, S = pca(img)

    im = U[:, 0]
    maxv = np.abs(im).max()
    im = im.reshape(shape) / maxv

    plt.subplot('121')
    plt.imshow(im, 'gray')

    im = U[:, 1]
    maxv = np.abs(im).max()
    im = im.reshape(shape) / maxv

    plt.subplot('122')
    plt.imshow(im, 'gray')

    plt.show()

if __name__ == '__main__':
    main()
