from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.misc import imresize, imsave

import os
import matplotlib.pyplot as plt
from math import floor, ceil
from skimage.filters import threshold_otsu

def cedar_load():
    PATH = '/home/victor/mestrado-local/bases/CEDAR55/'
    images = []
    for author in range(1, 56):
        for signature in range(1, 25):
            path = PATH + 'full_org/original_%d_%d.png' % (author, signature)
            images.append(np.array(Image.open(path).convert('L')))

    return images

def mcyt_load():
    PATH = '/home/victor/mestrado-local/bases/atvs/2004_MCYTDB_OffLineSignSubCorpus/'
    paths = []
    images = []
    for r in [x[2] for x in os.walk(PATH)][1:]:
        for l in r:
            if 'v' in l or 'V' in l:
                paths.append(l)

    for path in paths:
        image = Image.open('%s%s/%s' % (PATH, path[:4], path)).convert('L')
        images.append(np.array(image))

    return images

def sigcomp09_filename_for(author, PATH):
    genuines = list()
    forgeries = list()
    for file in os.listdir(PATH):
        if( (file[7:10] == '%03d' % author) and (file[11:14] == '%03d' % author) ):
            genuines.append(file)
        elif( (file[7:10] != '%03d' % author) and (file[11:14] == '%03d' % author) ):
            forgeries.append(file)

    return genuines, forgeries

def sigcomp09_load():
    PATH = '/home/victor/mestrado-local/bases/sigcomp2009-icdar/SigComp2009-training/NISDCC-offline-all-001-051-6g/'
    images = []

    for author in range(0, 51):
        genuines, forgeries = sigcomp09_filename_for(author+1, PATH)
        for file in genuines:
            images.append(np.array(Image.open(PATH+file).convert('L')))

    return images

def max_size(images):
    max_h = max_w = 0

    for image in images:
        h, w = image.shape

        if h > max_h:
            max_h = h

        if w > max_w:
            max_w = w

    return max_h, max_w

def otsu(image):
    threshold = threshold_otsu(image)
    image[np.where(image > threshold)] = 255

    return image

def invert(image):
    return 255 - image

def preprocess(images):
    max_h, max_w = max_size(images)

    center_h = ceil(max_h / 2)
    center_w = ceil(max_w / 2)
    preproc_images = []

    for image in images:
        preproc_image = np.ones((max_h, max_w), dtype='uint8') * 255

        image = otsu(image)

        center_mass_h, center_mass_w = ndimage.measurements.center_of_mass(image)
        height, width = image.shape

        center_mass_h = int(ceil(center_mass_h))
        center_mass_w = int(ceil(center_mass_w))

        # When either the height or width is equal to the max size, offset must be 0.
        offset_h = (center_h-center_mass_h) * (max_h != height)
        offset_w = (center_w-center_mass_w) * (max_w != width)

        preproc_image[offset_h:height+offset_h, offset_w:width+offset_w] = image
        preproc_image = invert(preproc_image)
        preproc_image = imresize(preproc_image, (256, 320))
        preproc_images.append(preproc_image)


    return preproc_images

def main():
    cedar_images = cedar_load()
    mcyt_images = mcyt_load()

    cedar_images = preprocess(cedar_images)
    mcyt_images = preprocess(mcyt_images)

    for k, image in enumerate(cedar_images):
        imsave('cedar-preprocessed/%d.png' % k, image)

    for k, image in enumerate(mcyt_images):
        imsave('mcyt-preprocessed/%d.png' % k, image)



if __name__ == '__main__':
    main()
