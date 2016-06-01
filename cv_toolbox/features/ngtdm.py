# encoding: utf-8

from core.util import get_histogram
import matplotlib.pyplot as plt
import numpy as np
import Image


class Ngtdm:
    def __init__(self, img, d):
        self.img = img
        self.d = d
        self.max_value = self.img.flatten().max()
        self.h, self.w = self.img.shape

        self.generate_matrix()
        self.coarseness()
        self.contrast()
        self.busyness()
        self.complexity()
        self.strength()
        self.features = {'cos' : self.cos, 'con' : self.con, 'bus' : self.bus, 'com' : self.com, 'str' : self.str}

    def generate_matrix(self):
        self.s = np.zeros(self.max_value + 1)
        for i in range(self.d, self.h - self.d):
            for j in range(self.d, self.w - self.d):

                window = self.img[
                    i - self.d:i + self.d + 1, j - self.d:j + self.d + 1].copy()

                window[self.d, self.d] = 0
                ai = np.sum(window.flatten())

                self.s[self.img[
                    i, j]] += np.abs(self.img[i, j] - (ai / float(np.power(2 * self.d + 1, 2) - 1)))

    def coarseness(self):

        self.n = (self.h - 2 * self.d) * (self.w - 2 * self.d)
        self.Ni = get_histogram(
            self.img[self.d:self.h - self.d, self.d:self.w - self.d],
            self.max_value)
        self.pi = self.Ni / self.n
        self.cos = np.sum(self.pi * self.s)**-1

    def contrast(self):
        Ng = np.sum(self.Ni > 0)

        first_term = 0

        for i in range(self.max_value + 1):
            for j in range(self.max_value + 1):
                first_term += self.pi[i] * self.pi[j] * (i - j)**2

        first_term = first_term / (Ng * (Ng - 1))
        second_term = self.s.sum() / self.n
        self.con = first_term * second_term

    def busyness(self):
        first_term = (self.pi * self.s).sum()

        second_term = 0

        for i in range(self.max_value + 1):
            for j in range(self.max_value + 1):
                if(self.pi[i] != 0 and self.pi[j] != 0):
                    second_term += abs(i * self.pi[i] - j * self.pi[j])

        self.bus = first_term / second_term

    def complexity(self):
        self.com = 0
        for i in range(self.max_value + 1):
            for j in range(self.max_value + 1):
                if(self.pi[i] != 0 and self.pi[j] != 0):
                    self.com += (abs(i - j) / (self.n * (self.pi[i] + self.pi[j]))) * (
                        self.pi[i] * self.s[i] + self.pi[j] * self.s[j])

    def strength(self):
        first_term = 0
        for i in range(self.max_value + 1):
            for j in range(self.max_value + 1):
                if(self.pi[i] != 0 and self.pi[j] != 0):
                    first_term += (self.pi[i] + self.pi[j]) * (i-j)**2

        second_term = self.s.sum()

        self.str = first_term / second_term

def main():
    img = np.array([[1, 1, 4, 3, 1], [3, 4, 0, 1, 1], [
                   5, 4, 2, 2, 2], [2, 1, 1, 4, 4], [0, 2, 2, 5, 1]])
    texture1 = np.array(Image.open('../images/texture1.tiff').convert('L'))
    texture2 = np.array(Image.open('../images/texture2.tiff').convert('L'))
    texture3 = np.array(Image.open('../images/texture3.tiff').convert('L'))
    texture4 = np.array(Image.open('../images/texture4.tiff').convert('L'))

    ngtdmt1 = Ngtdm(texture1, 1)
    ngtdmt2 = Ngtdm(texture2, 1)
    ngtdmt3 = Ngtdm(texture3, 1)
    ngtdmt4 = Ngtdm(texture4, 1)
    print ngtdmt1.features
    print ngtdmt2.features
    print ngtdmt3.features
    print ngtdmt4.features

if __name__ == '__main__':
    main()
