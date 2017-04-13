import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import bresenham
from scipy.misc import imsave, imresize
import math
import os
import os.path as P
import sys
from scipy import ndimage
from math import ceil


def invert(image):
    return 255 - image

def read(filename):
    points = []
    csvfile = open(filename, encoding="ISO-8859-1")
    reader = csv.reader(csvfile, delimiter=' ')
    for i in range(18):
        next(reader)

    min_x = min_y = np.inf
    max_x = max_y = 0

    for row in reader:
        if len(row) > 1:
            y = int(row[0])
            x = int(row[1])
            p = int(row[2])
            t = int(row[3])

            min_x = x if x < min_x else min_x
            min_y = y if y < min_y else min_y
            max_x = x if x > max_x else max_x
            max_y = y if y > max_y else max_y

            points.append({'x': x, 'y': y, 'p': p, 't': t})
        elif row[0] == '.PEN_DOWN':
            points.append({'x': x, 'y': y, 'p': -1, 't': t})

    return points, (min_x, max_x, min_y, max_y)


def preprocess(path, dimensions):

    target_im = np.array(Image.open('%s.tif' % (path)).convert('L'))
    points, (min_x, max_x, min_y, max_y) = read('%s.unp' % (path))
    online_im, _ = create_image(points, target_im.shape)
    
    max_h, max_w = dimensions

    center_h = ceil(max_h / 2)
    center_w = ceil(max_w / 2)

    center_mass_h, center_mass_w = ndimage.measurements.center_of_mass(target_im)
    height, width = target_im.shape

    center_mass_h = int(ceil(center_mass_h))
    center_mass_w = int(ceil(center_mass_w))

    # When either the height or width is equal to the max size, offset must be 0.
    offset_h = (center_h-center_mass_h) * (max_h != height) * (center_h>center_mass_h)
    offset_w = (center_w-center_mass_w) * (max_w != width) * (center_w>center_mass_w)


    preproc_target = np.zeros((max_h, max_w), dtype='uint8')
    preproc_online = np.zeros((max_h, max_w), dtype='uint8')
    # image = otsu(image)

    if offset_h < 5:
        offset_h = 0

    if offset_w < 5:
        offset_w = 0
        
    try:
        preproc_target[offset_h:height+offset_h, offset_w:width+offset_w] = invert(target_im)
        preproc_online[offset_h:height+offset_h, offset_w:width+offset_w] = online_im
    except Exception as e:
        import pdb; pdb.set_trace()

    # preproc_image = invert(preproc_image)
    preproc_target = imresize(preproc_target, (np.array(preproc_target.shape)*0.7).astype('int'))
    preproc_online = imresize(preproc_online, (np.array(preproc_online.shape)*0.7).astype('int'))
    # preproc_online = imresize(preproc_online, (155, 220))
    # preproc_target = imresize(preproc_target, (155, 220))

    return preproc_target, preproc_online


def create_image(points, shape, interpolate=True):
    size = len(points)
    all_points = []
    img_pressure = np.zeros(shape, dtype='float')
    img_speed = np.zeros(shape, dtype='float')
    for i in range(len(points) - 1):
        current_point, next_point = points[i], points[i + 1]

        if current_point['p'] > 0:
            all_points.append(current_point)

            if next_point['p'] > 0 and interpolate:
                x, y = current_point['x'], current_point['y']
                next_x, next_y = next_point['x'], next_point['y']
                p, next_p = current_point['p'], next_point['p']
                # s, next_s = current_point['s'], next_point['s']

                line_coordinates = bresenham.get_line((x, y), (next_x, next_y))
                pressures = np.linspace(p, next_p, len(line_coordinates))
                # speeds = np.linspace(p, next_p, len(line_coordinates))
                more_points = []

                for k, tuple in enumerate(line_coordinates):
                    # more_points.append({'x':tuple[0], 'y':tuple[1], 'p': int(pressures[k]), 's': speeds[k]})
                    more_points.append(
                        {'x': tuple[0], 'y': tuple[1], 'p': int(pressures[k])})

                for point in more_points:
                    all_points.append(point)

    for point in all_points:
        img_pressure[point['x'], point['y']] = point['p']
        # img_speed[point['x'], point['y']] += point['s']

    return img_pressure, img_speed


def add_velocity(points):
    size = len(points)
    speed = []
    max_s = 0
    min_s = np.inf
    for i in range(size - 1):
        current_i, next_i = i, i + 1
        if points[current_i]['p'] < 0:
            current_i = i - 1
        if points[next_i]['p'] < 0:
            next_i = i + 2

        x_i = points[current_i]['x']
        y_i = points[current_i]['y']
        x_i1 = points[next_i]['x']
        y_i1 = points[next_i]['y']

        displacement_i = math.sqrt((x_i1 - x_i)**2 + (y_i1 - y_i)**2)

        t_i = points[current_i]['t']
        t_i1 = points[next_i]['t']

        speed_i = displacement_i / (t_i1 - t_i)
        if min_s > speed_i:
            min_s = speed_i
        if max_s < speed_i:
            max_s = speed_i
        if speed_i < 0:
            print(speed_i)
        points[current_i]['s'] = speed_i

    points[size - 1]['s'] = points[size - 2]['s']

    return points, max_s, min_s


def main():
    PATH_LIST = [
                 '/home/vkslm/playground/datasets/IRONOFF-Subset/G/',
                 '/home/vkslm/playground/datasets/IRONOFF-Subset/F/',
                 '/home/vkslm/playground/datasets/IRONOFF-Subset/E/',
                 '/home/vkslm/playground/datasets/IRONOFF-Subset/C/',
                 '/home/vkslm/playground/datasets/IRONOFF-Subset/B_second/',
                 '/home/vkslm/playground/datasets/IRONOFF-Subset/B_first/',
                 ]
    # PATH_LIST = ['/home/victor/mestrado-local/bases/ironoff/Data/E/']

    global_min_x = global_min_y = np.inf
    global_max_x = global_max_y = 0
    # imgs_s = []
    imgs_path = []
    imgs_target = []

    global_max_x, global_max_y = (215, 548)
    initialized=True
    # min_s = np.inf
    # max_s = 0
    for PATH in PATH_LIST:
        for directory in os.listdir(PATH):
            for i in range(1, 27):
                try:
                    # points, max_i_s, min_i_s = add_velocity(points)
                    # if min_s > min_i_s: min_s = min_i_s
                    # if max_s < max_i_s: max_s = max_i_s
                    path = PATH + directory + '/%s.champs%d' % (directory, i)
                    # import pdb; pdb.set_trace()
                    
                    if P.isfile(path+'.tif'):
                        imgs_path.append(path)  
                        points, (min_x, max_x, min_y, max_y) = read(
                            PATH + directory + '/%s.champs%d.unp' % (directory, i))
                        
                    if not initialized:
                        offline_target = np.array(Image.open(
                            PATH + directory + '/%s.champs%d.tif' % (directory, i)))
                        h, w = offline_target.shape
                        max_x = h if h > max_x else max_x
                        max_y = w if w > max_y else max_y
                        
                        global_max_x = max_x if max_x > global_max_x else global_max_x
                        global_max_y = max_y if max_y > global_max_y else global_max_y
                        # img_p, img_s = create_image(points, shape)
                        print (global_max_x, global_max_y)
                    # shape = offline_target.shape


                    # imgs_s.append(img_s)
                    # imgs_p.append(img_p)
                    # imgs_target.append(offline_target)
                except Exception as e:
                    print('Error with: ' + PATH + directory +
                          '/%s.champs%d' % (directory, i))
                    print(e)
                    # sys.exit()
    print (global_max_x, global_max_y)

    # print(min_s, max_s)
    # for img_s in imgs_s:
        # img_s = 255 * (img_s/max_s)

    # print(imgs_p[0].max())
    # import pdb; pdb.set_trace()
    # offline images are .tif and online date is .unp
    for i in range(len(imgs_path)):
        # imgs_p[i] = 255 * (imgs_p[i]/imgs_p[i].max())
        path = imgs_path[i]
        print(path)
        offline, online = preprocess(path, (global_max_x, global_max_y))

        imsave('/home/vkslm/playground/datasets/IRONOFF-Subset/out/%010d-on.png' % i, online)
        # imsave('out/on-%i-s.png' % i, imgs_s[i][global_min_x-10:global_max_x+10, global_min_y-10:global_max_y+10])
        imsave('/home/vkslm/playground/datasets/IRONOFF-Subset/out/%010d-off.png' % i, offline)

        # imsave('out/on-%i.png' % i,
        #     np.array(Image.open('out/on-%i.png' % i).convert('L')))

        # imsave('out/off-%i.png' % i,
        # np.array(Image.open('out/off-%i.png' %
        # i).convert('L'))[global_min_x-10:global_max_x+10,
        # global_min_y-10:global_max_y+10])

        # imsave('out/on-%i-s.png' % i,
        # np.array(Image.open('out/on-%i-s.png' %
        # i).convert('L'))[global_min_x-10:global_max_x+10,
        # global_min_y-10:global_max_y+10])


if __name__ == '__main__':
    main()
