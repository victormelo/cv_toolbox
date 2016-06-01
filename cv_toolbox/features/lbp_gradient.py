# encoding: utf-8

from ..core.util import gradient
import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image

def lbp_histogram(img, p, r):
    lbp = local_binary_pattern(img, p, r, method='uniform').astype(np.uint8)
    n_bins = lbp.max() + 1
    hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
    return hist, lbp


def lbp_gradient(img, p, r):
    img, _ = gradient(img)
    hist, lbp = lbp_histogram(img, p, r)
    return hist


def main():
    img = np.array(Image.open('images/2-2-g-gray.PNG').convert('L')).astype('float')
    print glbp(img).shape


if __name__ == '__main__':
    main()
