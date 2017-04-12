import csv
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from PIL import Image
import bresenham
from scipy.misc import imsave
import math
import os
import sys

def read(filename):
    points = []
    reader = sio.loadmat(filename)

    for key in range(reader['x'].shape[1]):
        x = np.round(reader['x'][0, key] / 13731 * (210-1))
        y = np.round(reader['y'][0, key] / 20504 * (545-1))
        p = np.round(reader['p'][0, key] / 1024 * 255)
        points.append({'x' : int(x), 'y': int(y), 'p': int(p)})

    return points

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
                    more_points.append({'x':tuple[0], 'y':tuple[1], 'p': int(pressures[k])})

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
        if points[current_i]['p'] < 0: current_i = i-1
        if points[next_i]['p'] < 0: next_i = i+2

        x_i = points[current_i]['x']
        y_i = points[current_i]['y']
        x_i1 = points[next_i]['x']
        y_i1 = points[next_i]['y']

        displacement_i = math.sqrt((x_i1 - x_i)**2 + (y_i1 - y_i)**2)

        t_i = points[current_i]['t']
        t_i1 = points[next_i]['t']

        speed_i = displacement_i / (t_i1 - t_i)
        if min_s > speed_i: min_s = speed_i
        if max_s < speed_i: max_s = speed_i
        if speed_i < 0: print(speed_i)
        points[current_i]['s'] = speed_i

    points[size-1]['s'] = points[size-2]['s']

    return points, max_s, min_s

def main():
    PATH_LIST = ['/home/victor/mestrado-local/bases/ironoff/Data/F/',
             '/home/victor/mestrado-local/bases/ironoff/Data/E/']
    # PATH_LIST = ['/home/victor/mestrado-local/bases/ironoff/Data/E/']

    global_min_x = global_min_y = np.inf
    global_max_x = global_max_y = 0
    imgs_s = []
    imgs_p = []
    imgs_target = []

    for PATH in PATH_LIST:
        for directory in os.listdir(PATH):
            for i in range(1, 27):
                try:
                    points = read(PATH+directory+'/%s.champs%d.unp' % (directory, i))
                    # points, max_i_s, min_i_s = add_velocity(points)
                    # if min_s > min_i_s: min_s = min_i_s
                    # if max_s < max_i_s: max_s = max_i_s

                    offline_target = np.array(Image.open(PATH+directory+'/%s.champs%d.tif' % (directory, i)).convert('L'))
                    shape = offline_target.shape

                    img_p, img_s = create_image(points, shape)

                    # imgs_s.append(img_s)
                    imgs_p.append(img_p)
                    imgs_target.append(offline_target)
                except Exception as e:
                    print('Error with: ' + PATH+directory+'/%s.champs%d' % (directory, i))
                    print(e)
                    # sys.exit()

    # print(min_s, max_s)
    # for img_s in imgs_s:
        # img_s = 255 * (img_s/max_s)

    # print(imgs_p[0].max())
    # exit()

    for i in range(len(imgs_p)):
        # imgs_p[i] = 255 * (imgs_p[i]/imgs_p[i].max())
        imsave('out/on-%010d.png' % i, (imgs_p[i])[global_min_x:global_max_x, global_min_y:global_max_y])
        # imsave('out/on-%i-s.png' % i, imgs_s[i][global_min_x-10:global_max_x+10, global_min_y-10:global_max_y+10])
        imsave('out/off-%010d.png' % i, (255-imgs_target[i])[global_min_x:global_max_x, global_min_y:global_max_y])

        # imsave('out/on-%i.png' % i,
        #     np.array(Image.open('out/on-%i.png' % i).convert('L')))

        # imsave('out/off-%i.png' % i,
        #     np.array(Image.open('out/off-%i.png' % i).convert('L'))[global_min_x-10:global_max_x+10, global_min_y-10:global_max_y+10])

        # imsave('out/on-%i-s.png' % i,
        #     np.array(Image.open('out/on-%i-s.png' % i).convert('L'))[global_min_x-10:global_max_x+10, global_min_y-10:global_max_y+10])


if __name__ == '__main__':
    main()
