# encoding: utf-8

from core.util import get_histogram
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import cv2
import Image

def eqhist(img):
    h, w = img.shape
    size = h * w

    img_return = np.zeros((h, w))

    higher_level = 255

    # coloca os valores de cada escala de cinza no histograma
    histogram = get_histogram(img, higher_level)

    # calcula-se as probabilidades de cada nível de cinza
    probs = histogram / size

    # calcula os novos valores de cada tom de cinza
    new_values = np.cumsum(probs) * higher_level

    # altera na imagem os novos valores de cinza na imagem
    for i in range(h):
        for j in range(w):
            img_return[i, j] = new_values[img[i, j]]

    return img_return


def main():
    img = np.array(Image.open('../images/figura1.jpg').convert('L'))

    eq = eqhist(img)

    fig = plt.figure()

    fig_1 = fig.add_subplot(1, 2, 1)
    fig_1.imshow(img, cmap='gray')

    fig_2 = fig.add_subplot(1, 2, 2)
    fig_2.imshow(eq, cmap='gray')

    plt.show()


if __name__ == '__main__':
    main()
