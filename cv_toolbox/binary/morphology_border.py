from skimage.filters import threshold_otsu
from skimage.morphology import erosion
import numpy as np
import Image
import matplotlib.pyplot as plt

def main():
    img = np.array(Image.open('../images/1-1-g.PNG').convert('L'))
    threshold = threshold_otsu(img)
    imbw = img >= threshold

    imerosion = erosion(imbw)
    plt.imshow(img, 'gray')

    plt.figure()
    plt.imshow(imbw, 'gray')

    plt.figure()
    plt.imshow((imerosion ^ imbw), 'gray')
    plt.show()

if __name__ == '__main__':
    main()
