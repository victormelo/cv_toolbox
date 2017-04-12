from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from skimage.filters import gaussian
from math import sqrt
import scipy.stats as st
from math import sqrt
from scipy.signal import convolve2d

def fspecial_gauss(size, sigma, amp):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    g = np.exp( - ( (x**2 + y**2) / (2.0*sigma**2) ) )

    return amp * g/g.sum()

def apply_function(img, pixels, sigma):
    radius  = 1
    # imgPressures = imgPressures / (imgPressures.max()/2.2) + 0.2
    new_image = np.zeros(img.shape)

    k = 0
    for pixel in pixels:
        (j, i, p) = pixel
        window = img[(i-radius):(i+radius+1) , (j-radius):(j+radius+1)]
        kernel = fspecial_gauss(3, sqrt(sigma), p)

        blurred = convolve2d(window, kernel)
        convoW, convoH = blurred.shape
        centerW = convoW // 2
        centerH = convoH // 2
        new_image[i-centerW:i+centerW+1,j-centerH:j+centerH+1] = blurred
        k+=1
        print(k)

    from scipy.ndimage.filters import gaussian_filter

    new_image = gaussian_filter(new_image, sigma=2)

    new_image = new_image / (new_image.max()/0.5)

    return new_image




if __name__ == '__main__':
    imagem = np.array([ [255,0,255,0,255,0,0,255,255,255,0],
                            [0,255,0,0,0,0,0,0,0,0,0],
                            [0,0,255,0,0,0,0,0,0,0,0],
                            [0,0,0,255,255,0,0,0,0,0,0],
                            [0,0,0,0,0,255,255,0,0,0,0],
                            [0,0,0,0,0,0,255,255,0,0,0],
                            [0,0,0,0,0,0,0,0,255,0,0],
                            [0,0,0,0,0,0,0,0,255,0,0],
                            [0,0,0,0,0,0,0,0,255,0,0],
                            [0,0,0,0,0,0,0,0,255,0,0]])

    r = gaussian_filter(imagem, 708.6)
    plt.imshow(r, 'gray', interpolation='none')
    plt.figure()
    plt.imshow(imagem, 'gray', interpolation='none')
    plt.show()
