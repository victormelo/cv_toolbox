"""Verify a signature
Usage:
    prep.py <dataset_folder> <info_fn>

"""

from docopt import docopt
from skimage.morphology import skeletonize_3d, thin
import cv2
import glob
import os
import os.path as P
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def preprocess(path, info_fn):
    with open(info_fn) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for person, signum in zip(content[::2], content[1::2]):
        signum = signum.split(' ')
        ngenuines, nforgeries = signum
        ngenuines = int(ngenuines.replace("g:", ""))
        nforgeries = int(nforgeries.replace("f:", ""))
        genuines = []
        forgeries = []
        for gn in range(1, ngenuines+1):
            f = "g-%03d.png" % gn
            fn = P.join(P.join(path, person), f)
            im = cv2.imread(fn, 0)
            im = np.abs(im - 255)
            skel = thin(im)

            plt.imshow(skel, 'gray')

            plt.show()
            plt.figure()
        # for fg in range(1, xnforgeries+1):
        #     f = "f-%03d.png" % fg
        #     fn = P.join(P.join(path, person), f)
        #     im = cv2.imread(fn)

def main(args):
    print(args)
    preprocess(args['<dataset_folder>'], args['<info_fn>'])

if __name__ == "__main__":
    main(docopt(__doc__))