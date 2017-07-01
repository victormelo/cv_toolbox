import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
import sys

def to_rgb(im):
    # I think this will be slow
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 0] = im
    ret[:, :, 1] = im
    ret[:, :, 2] = im
    return ret

class Annotate(object):
    def __init__(self, img, filename):
        self.ax = plt.gca()
        self.x0 = None
        self.y0 = None
        self.x1 = None
        self.y1 = None
        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_press_key)
        self.img = img
        self.filename = filename

    def on_press_key(self, event):
        print(('press', event.key))
        if event.key == 'r':
            plt.imshow(self.img)
            plt.draw()
        elif event.key =='alt+s':
            plt.imsave('result-'+self.filename, self.imgcp)

    def on_press(self, event):
        print ('press')
        self.x0 = event.xdata
        self.y0 = event.ydata

    def on_release(self, event):
        print ('release')
        self.imgcp = np.array(self.img).copy()

        self.x1 = event.xdata
        self.y1 = event.ydata
        # npimg = np.array(img)
        # print (npimg.shape)
        y0 = int(min(self.x0, self.x1))
        y1 = int(max(self.x0, self.x1))
        x0 = int(min(self.y0, self.y1))
        x1 = int(max(self.y0, self.y1))

        crop = np.array(Image.fromarray(self.imgcp[x0:x1, y0:y1]).convert('L'))
        t = threshold_otsu(crop)

        self.imgcp[:,:,3] = 0
        self.imgcp[x0:x1, y0:y1][np.where(crop<=t)] = (255, 0, 0, 255)
        plt.clf()

        plt.imshow(self.imgcp)
        plt.draw()


def run(filename):
    img = Image.open(filename).convert('RGBA')
    a = Annotate(img, filename)
    plt.imshow(img, 'gray')
    plt.show()

if __name__ == '__main__':
    if(sys.argv[1]):
        run(sys.argv[1])

