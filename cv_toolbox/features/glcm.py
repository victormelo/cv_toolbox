# encoding: utf-8

from core.util import get_histogram
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix
import numpy as np
from PIL import Image
from skimage.exposure import adjust_log

def glcm(matrix, offset):
    h, w = matrix.shape
    di, dj = offset
    minv, maxv = matrix.min(), matrix.max()
    co_oc = np.zeros((maxv + 1, maxv + 1), dtype='int')

    for i in range(h):
        for j in range(w - 1):
            transition = matrix[i, j], matrix[i, j + 1]
            co_oc[transition] += 1

    return co_oc


def glcm_features(co_oc):
    co_oc = co_oc.astype('float')
    p = co_oc / co_oc.sum()
    h, w = co_oc.shape
    max_prob = p.max()
    j, i = np.meshgrid(np.arange(0, w), np.arange(0, h))
    mr = (np.arange(h) * p.sum(axis=1)).sum()
    mc = (np.arange(w) * p.sum(axis=0)).sum()
    sigmar = ((np.arange(h) - mr)**2 * p.sum(axis=1)).sum()
    sigmac = ((np.arange(w) - mc)**2 * p.sum(axis=0)).sum()
    dpr = np.sqrt(sigmar)
    dpc = np.sqrt(sigmac)

    contrast = ((i - j)**2 * p).sum()

    homogeneity = (p / (1 + np.abs(i - j))).sum()

    correlation = (
        ((i - mr) * (j - mc) * p) / (dpr * dpc)).sum() if dpr != 0 and dpc != 0 else 0

    energy = np.sum(p**2)
    entropy = np.sum(p[p != 0] * -np.log(p[p != 0]))

    return {'max_prob': max_prob, 'contrast': contrast,
            'homogeneity': homogeneity, 'energy': energy,
            'entropy': entropy, 'correlation': correlation}


def main():
    figa = np.array(Image.open('images/Fig1130(a)(uniform_noise).tif').convert('L'))
    figb = np.array(Image.open('images/Fig1130(b)(sinusoidal).tif').convert('L'))
    figc = np.array(Image.open('images/Fig1130(c)(cktboard_section).tif').convert('L'))

    glcma = glcm(figa, (0, 1))
    glcmb = glcm(figb, (0, 1))
    glcmc = glcm(figc, (0, 1))
    fa = glcm_features(glcma)
    fb = glcm_features(glcmb)
    fc = glcm_features(glcmc)

    ax = plt.subplot('321')
    ax.set_title("Figure a")
    ax.imshow(figa, 'gray')
    ax.axis('off')

    ax = plt.subplot('322')
    ax.set_title("GLCM of Figure b")
    ax.imshow(adjust_log(glcma), 'gray')
    ax.axis('off')

    ax = plt.subplot('323')
    ax.set_title("Figure b")
    ax.imshow(figb, 'gray')
    ax.axis('off')

    ax = plt.subplot('324')
    ax.set_title("GLCM of Figure b")
    ax.imshow(adjust_log(glcmb), 'gray')
    ax.axis('off')

    ax = plt.subplot('325')
    ax.set_title("Figure c")
    ax.imshow(figc, 'gray')
    ax.axis('off')

    ax = plt.subplot('326')
    ax.set_title("GLCM of Figure c")
    plt.imshow(adjust_log(glcmc), 'gray')
    ax.axis('off')

    plt.show()

    # print 'Figure a\n', fa
    # print 'Figure b\n', fb
    # print 'Figure c\n', fc


def sglcm(matrix):

    gmatrix = greycomatrix(matrix, [1], [0, -np.pi/4, -np.pi/2, -3*np.pi/4], symmetric=False, normed=True)

    g1 = glcm_features(gmatrix[:,:,0,0])
    g2 = glcm_features(gmatrix[:,:,0,1])
    g3 = glcm_features(gmatrix[:,:,0,2])
    g4 = glcm_features(gmatrix[:,:,0,3])

    H = np.array([g1['homogeneity'], g2['homogeneity'], g3['homogeneity'], g4['homogeneity']])
    C = np.array([g1['contrast'], g2['contrast'], g3['contrast'], g4['contrast']])
    E = np.array([g1['entropy'], g2['entropy'], g3['entropy'], g4['entropy']])
    O = np.array([g1['correlation'], g2['correlation'], g3['correlation'], g4['correlation']])

    M = np.array([np.mean(H), np.mean(C), np.mean(E), np.mean(O)])
    S = np.array([np.std(H), np.std(C), np.std(E), np.std(O)])
    R = np.array([np.max(H) - np.min(H), np.max(C) - np.min(C), np.max(E) - np.min(E), np.max(O) - np.min(O)])
    sglcm = np.array([M, S, R]) / (M + S + R)
    return sglcm.flatten()


if __name__ == '__main__':
    test()
