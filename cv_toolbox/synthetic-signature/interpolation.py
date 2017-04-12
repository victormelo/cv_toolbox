import csv
import numpy as np
import matplotlib.pyplot as plt
import bresenham
from skimage.morphology import dilation, square
import scipy
import scipy.io
from gauss import apply_function

def read_file(filename, delimiter=' ', skip=()):
    points = {'x' : [], 'y': [], 'p': []}
    if(filename.endswith('.HWR')):
        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)

            # ignore two first lines
            # reader.next()
            # reader.next()
            for r in reader:
                row = [ x for x in r if x.isdigit() ]
                points['x'].append(row[0])
                points['y'].append(row[1])
                points['p'].append(row[2])
    elif (filename.endswith('.hwr')):

        with open(filename) as csvfile:
            reader = csv.reader(csvfile, delimiter=delimiter)
            reader.next()

            for r in reader:
                row = filter(None, r)
                if(row[0] == '\t'):
                    row = row[1:]
                else:
                    row[0] = row[0].replace('\t', '')
                points['x'].append(int(row[1]))
                points['y'].append((row[0]))
                points['p'].append((row[2]))

    elif (filename.endswith('.mat')):

        mat = scipy.io.loadmat(filename)
        points['x'] = mat['x'].flatten()
        points['y'] = mat['y'].flatten()
        points['p'] = mat['p'].flatten()

    elif (filename.endswith('.unp')):
        csvfile = open(filename)
        reader = csv.reader(csvfile, delimiter=delimiter)
        for i in range(18): next(reader)

        for row in reader:
            if len(row) > 1:
                points['x'].append(row[0])
                points['y'].append(row[1])
                points['p'].append(row[2])


    return points


def create_image(shape, xs, ys, ps, interpolate=True):

    img_s = np.ones(shape)*255
    img_pressures = np.ones(shape)
    img_spu = np.ones(shape)
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

        # if p == 0:
            # img_spu[yp, xp] = 1

        if p > 0:
            # img_pressures[yp, xp] = p
            img_s[xp-2:xp+2, yp-2:yp+2] = 255-p

    return img_s, img_spu, img_pressures, pixels

def main():
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    points = read_file('nisdcc/NISDCC-001_001_003.hwr')
    # points = read_file('002_1.HWR')

    rtab = 2540.0
    rscan = 600.0 # for Sigcomp
    # rscan = 400.0 # for Biosecurid

    frequency_rate = 200.0 # 200 Hz for sigcomp

    # frequency_rate = 100.0 # 100 Hz for biosecurid

    factor = rscan / rtab
    print(factor)
    xs = np.array(points['x']).astype('int')  * factor
    ys = np.array(points['y']).astype('int')  * factor

    xs = (xs.max() - xs).astype('int')
    ys = ys.astype('int')

    ps  = np.array(points['p']).astype('int')
    size = len(xs)
    period = 1 / frequency_rate
    total_time = size * period
    # import pdb; pdb.set_trace()
    time = np.arange(0, total_time, period)

    speed = []

    for i in range(size-1):
        x_i = xs[i]
        y_i = ys[i]
        x_i1 = xs[i+1]
        y_i1 = ys[i+1]

        t_i = time[i]
        t_i1 = time[i+1]

        sx = x_i1 - x_i
        sy = y_i1 - y_i

        st = (t_i1 - t_i)
        vx = sx / st
        vy = sy / st

        speed.append((vx, vy))

    speed.append(speed[size-2])


    h = int(xs.max()) + 50
    w = int(ys.max()) + 50

    phiPen = 3
    delta = 2.54
    phi = phiPen * (rscan/delta)


    # img_spu = dilation(img_spu, selem)


    img_s, img_spu, img_pressures, pixels  = create_image((h, w), xs, ys, ps)

    img_pressures = img_pressures/ (1024/2.2) + 0.2

    t = 0
    img_s_new = np.zeros(img_s.shape)
    pixel_normalized = []
    for pixel in pixels:
        (j, i, p) = pixel
        # p = img_pressures[, j]
        pixel = (j, i, p)
        pixel_normalized.append(pixel)

    # img_s = apply_function(img_s, pixel_normalized, phi)



    # selem = square(3)
    # import pdb; pdb.set_trace()
    # img_s = dilation(img_s, selem)


    # scipy.misc.imsave('is.png', img_s.max() - img_s)
    # exit()

    # img_spu = dilation(img_spu, selem)




    non_empty = np.where(img_s > 0)
    cropBox = (min(non_empty[0]), max(non_empty[0]), min(non_empty[1]), max(non_empty[1]))
    img_new_s = img_s[cropBox[0]:cropBox[1], cropBox[2]:cropBox[3]]
    # img_new_spu = img_spu[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
    # img_pressures = img_pressures[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]



    # img_new_s = np.flipud(img_new_s)
    # img_new_spu = np.flipud(img_new_spu)
    # img_pressures = np.flipud(img_pressures)


    # exit()

    # scipy.misc.imsave('is.png', img_new_s.max() - img_new_s)
    print(img_new_s.shape)
    scipy.misc.imsave('is.png', img_s)
    # scipy.misc.imsave('ispu.png', img_new_spu)
    # plt.plot(ys, xs)
    plt.imshow(img_s)
    plt.show()
    # import pdb; pdb.set_trace()

    # print p.max()
    # print size

if __name__ == '__main__':
    main()

