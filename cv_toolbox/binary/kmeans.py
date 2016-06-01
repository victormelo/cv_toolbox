# encoding: utf-8

from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import Image

class KMeans():
    def __init__(self, img, k=2):
        self.k = k
        self.img = img

    def run(self):

        h, w = self.img.shape
        iterations = 10
        self.centroids = np.random.uniform(self.img.min(), self.img.max(), self.k).astype('int')

        for i in range(iterations):
            self.points = []
            for x in range(self.k):
                self.points.append([])

            for i in range(h):
                for j in range(w):
                    pixel = self.img[i, j]
                    group = self.get_group(pixel)
                    self.points[group].append(pixel)

            self.recalculate_centroid()

    def get_group(self, pixel):
        return np.argmin(distance(self.centroids, pixel))

    def new_image(self):
        h,w = self.img.shape
        retorno = np.zeros((h,w))
        for i in range(h):
                for j in range(w):
                    retorno[i,j] = self.centroids[self.get_group(self.img[i,j])]

        return retorno

    def recalculate_centroid(self):
        self.centroids = []

        for point in self.points:
            mean = np.array(point).mean().astype('int')
            self.centroids.append(mean)


def distance(a, b):
    return np.sqrt(np.power(a-b, 2))


def main():
    img = np.array(Image.open('../../images/kmeans.png').convert('L'))
    img = np.array(Image.open('../../images/figura1.jpg').convert('L'))
    from cv_toolbox.preproccess.eqhist import eqhist

    img = eqhist(img)
    kmeans = KMeans(img)
    kmeans.run()

    img_b = kmeans.new_image()

    kmeans.k = 4
    kmeans.run()
    img_b4 = kmeans.new_image()

    kmeans.k = 8
    kmeans.run()
    img_b8 = kmeans.new_image()

    plt.imshow(np.hstack((img, img_b, img_b4, img_b8)), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
