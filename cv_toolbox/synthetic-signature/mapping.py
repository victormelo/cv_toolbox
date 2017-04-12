from interpolation import read_file, create_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import scipy.misc
from skimage.morphology import skeletonize, binary_erosion, disk, binary_dilation
from skimage.filters import threshold_otsu
import bresenham


def to_rgb(im):
    # I think this will be slow
    w, h = im.shape

    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

def map_onoff(off, xs, ys, ps, interpolate=False):

    size = len(xs)
    pixels = []
    for i in range(size):
        x = int(xs[i])
        y = int(ys[i])
        p = int(ps[i])
        pixels.append((x, y, p))


        if i < size-1:
            next_x = int(xs[i+1])
            next_y = int(ys[i+1])
            next_p = int(ps[i+1])

            if interpolate:
                line_coordinates = bresenham.get_line((x, y), (next_x, next_y))
                pressures = np.linspace(p, next_p, len(line_coordinates))
                more_pixels = []

                for k, tuple in enumerate(line_coordinates):
                    more_pixels.append(tuple + (pressures[k], ) )

                for pixel in more_pixels:
                    pixels.append(pixel)


    for pixel in pixels:
        (xp, yp, p) = pixel

        if p > 0:
            off[yp, xp] = [255-p, 255-p, 255-p]

    return off

if __name__ == '__main__':
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description='Map an online image to its respective offline.')
    parser.add_argument('filename', nargs=1, metavar='FILENAME', type=str, help='Filename without extension')

    args = parser.parse_args()
    filename = args.filename[0]
    img_off = np.array(Image.open(filename+'.tif').convert('L'))
    img_off = to_rgb(img_off)

    on = read_file(filename+'.unp')
    xs = np.array(on['x']).astype('int')
    ys = np.array(on['y']).astype('int')
    ps  = np.array(on['p']).astype('int')

    result = map_onoff(np.ones(img_off.shape)*255, xs, ys, ps, interpolate=False)
    scipy.misc.imsave('results/'+filename.replace('/', '')+'.png', result)
    # plt.imshow(result)

