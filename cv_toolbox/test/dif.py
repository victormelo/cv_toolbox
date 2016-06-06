import matplotlib.pyplot as plt
import numpy as np

def difference(gray, gt):
    # print .shape
    sig = np.logical_not(( gt[:, :, 0] - gray  ) > 0)*255

    dif = gray - sig
    # dif = dif[dif < 0]  = 0
    dif[np.where( dif <= 0 )] = 255
    # plt.imshow(gray, 'gray'); plt.show()
    plt.imshow(dif, 'gray'); plt.show()
    # plt.imshosw(gray[780:808, 190:510], 'gray'); plt.show()
    # plt.imshow(gray  , 'gray'); plt.show()
    # return gray - gt

if __name__ == '__main__':
    import argparse
    from PIL import Image

    parser = argparse.ArgumentParser(description='Difference between two images.')
    parser.add_argument('-d', type=str, nargs=2, metavar=('im1', 'im2'), help='the two images')

    args = parser.parse_args()
    gray = np.array(Image.open(args.d[0]).convert('L'))
    gt = np.array(Image.open(args.d[1]))
    difference(gray, gt)
    # print(args.accumulate(args.integers))
