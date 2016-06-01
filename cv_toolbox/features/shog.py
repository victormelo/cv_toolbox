# encoding: utf-8

from ..core.util import gradient, grid_image
import numpy as np
from PIL import Image
from math import floor
from sklearn import preprocessing
import matplotlib.pyplot as plt

def shog(img):
    # 1 - Calculate first-order gradient image GI (x, y) for an input image,
    # then direction feature for every pixel.
    magnitude, angles = gradient(img)

    # convert angles to degrees between 0-180 (non-directional)
    # angles =  angles * (180 / np.pi) % 180

    # 2 - Segment the signature image into blocks. The image is divided into zones using a
    # fixed number of rectangular grids in Cartesian coordinate. Divide the whole signature
    # into 6 blocks with equal size, while the red rectangles show the
    # overlapped blocks for feature extraction.
    grid_coordinates = grid_image(img, 2, 3, 0.25)

    # 3 - Compute one histogram of directional gradient for each block. The histogram is
    # quantified into 9 bins between 0-180 (non-directional), and the sum of gradient magnitude
    # within one block is computed. After that, normalization is implemented by L2-Norm, thus
    # each block has one feature histogram.
    B = 9
    h, w = img.shape
    # exit()

    nwin_x=2;
    nwin_y=3;

    step_x = int(floor(h / (nwin_x+1)))
    step_y = int(floor(w / (nwin_y+1)))
    angs = np.sort(np.linspace(-np.pi+2*np.pi/B, np.pi, B))
    hog = np.array([])

    for coord in grid_coordinates:
        x, xend, y, yend = coord
        angles_grid = angles[x:xend, y:yend].flatten()
        magnitude_grid = magnitude[x:xend, y:yend].flatten()
        K = len(angles_grid)
        current_bin = 0
        hist_grid = np.zeros(B)
        for ang_lim in angs:
            for k in range(K):
                if angles_grid[k] < ang_lim:
                    angles_grid[k] = np.inf
                    hist_grid[current_bin] = hist_grid[current_bin] + magnitude_grid[k]
            current_bin += 1

        # 4 - Final feature vector is achieved by concatenating six histograms obtained
        hist_grid_normalized = preprocessing.normalize(hist_grid.reshape(1,-1), norm='l2')
        hog = np.append(hog, hist_grid_normalized)

    return hog

def main():
    # img = np.array(Image.open('images/2-2-g.PNG').convert('L')).astype('float')
    # import matplotlib.pyplot as plt
    # plt.imsave('images/2-2-g-gray.PNG', img, cmap='gray')
    img = np.array(Image.open('images/2-2-g-gray.PNG')).astype('float')
    img = img[:, :, 0]
    hog = shog(img)
    print hog

if __name__ == '__main__':
    main()
