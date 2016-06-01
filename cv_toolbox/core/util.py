# -*- coding: utf-8 -*-

from skimage import io
from skimage.color import rgb2gray
import numpy as np
from math import floor


def grid_image(img, h_grid, w_grid, overlap):
    h, w = img.shape
    h_factor = floor(h / h_grid * (1 + overlap))
    w_factor = floor(w / w_grid * (1 + overlap))
    coordinates = list()

    for i in range(h_grid):
        for j in range(w_grid):
            x = floor(i*h_factor*(1-overlap))
            y = floor(j*w_factor*(1-overlap))
            coordinates.append((int(x), int(x+h_factor), int(y), int(y+w_factor)))

    return coordinates

def grayscale(img):
    try:
        return rgb2gray(img)
    except:
        return img


def get_histogram(img, higher_level):
    h, w = img.shape

    histogram = np.zeros(higher_level + 1)

    for i in range(h):
        for j in range(w):
            histogram[img[i, j]] = histogram[img[i, j]] + 1

    return histogram


def gradient(img):
    img = np.array(img).astype('float')
    gx = np.empty(img.shape, dtype=np.double)
    gx[:, 0] = 0
    gx[:, -1] = 0
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]

    gy = np.empty(img.shape, dtype=np.double)
    gy[0, :] = 0
    gy[-1, :] = 0
    gy[1:-1, :] = img[2:, :] - img[:-2, :]

    g = np.sqrt(gx**2 + gy**2)
    orientations = np.arctan2(gx, gy)
    return g, orientations

# def get_pixel(img, i, j):
#     h, w = img.shape
#     if(i >= h-1 or j >= w-1 or i < 0 or j < 0):
#         return 1

#     return img[i, j]

def main():
    import Image
    import matplotlib.pyplot as plt
    import cv2
    from skimage.filters.rank import gradient as g
    from skimage import io, data

    img = np.array(Image.open('images/2-2-g.PNG').convert('L')).astype('float')
    g, orientations = gradient(img)
    plt.imshow(g, 'gray')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
