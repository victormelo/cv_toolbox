from PIL import Image
from cedar55 import load_database
import matplotlib.pyplot as plt


def train():
    pass


def main():
    authors = load_database()

def show(author):
    ax = plt.subplot('231')
    ax.imshow(author['genuine'][0], 'gray')
    ax.axis('off')

    ax = plt.subplot('232')
    ax.imshow(author['genuine'][1], 'gray')
    ax.axis('off')

    ax = plt.subplot('233')
    ax.imshow(author['genuine'][2], 'gray')
    ax.axis('off')

    ax = plt.subplot('234')
    ax.imshow(author['forgeries'][0], 'gray')
    ax.axis('off')

    ax = plt.subplot('235')
    ax.imshow(author['forgeries'][1], 'gray')
    ax.axis('off')

    ax = plt.subplot('236')
    ax.imshow(author['forgeries'][2], 'gray')
    ax.axis('off')

    plt.show()


if __name__ == '__main__':
    main()
