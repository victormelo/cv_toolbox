import csv
import numpy as np
import matplotlib.pyplot as plt

def read_csv(filename, delimiter=' ', skip=()):
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)

        # ignore two first lines
        reader.next()
        reader.next()

        points = []
        for row in reader:
            points.append({'x': float(row[0]), 'y': float(row[1]), 't': float(row[2]), 'p': float(row[3]), 'endpts': int(row[4]) })

    return points

def main():
    points = read_csv('/home/victor/mestrado-local/bases/SUSig/VisualSubCorpus/GENUINE/SESSION1/004_1_3.sig')
    # points = read_csv('data/with-user/elgs@ecomp.poli.br/genuine/1/1.data')
    xs = np.array([point['x'] for point in points]).astype('int')
    ys = np.array([point['y'] for point in points]).astype('int')
    p = (np.array([point['p'] for point in points]) / 1023 * 255).astype('int')
    plt.plot(xs, -ys)
    plt.show()
    # print p.max()
    # size = len(xs)
    # print size

    # w = 10800
    # h = 6480

    # img = np.zeros((h, w))
    # print(p.max())
    # for i in range(size):
    #     if(img[xs[i], ys[i] += p[i]
    # import scipy.misc
    # scipy.misc.imsave('outfile.jpg', 255-img)
    # # plt.imshow(img, cmap='gray')
    # plt.plot(ys, xs)
    # plt.show()



if __name__ == '__main__':
    main()
