from PIL import Image
import numpy as np
from feature import extract_feature_vector

PATH = '/home/victor/mestrado-local/bases/CEDAR55/'

def load_dataset():
    train = []
    test = []
    validation = []
    for i in range(55):
        g, f = load_feature_vector(i)
        indices = np.random.permutation(24)
        indices = np.arange(24)
        train.append({'g': g[indices[0:8]], 'f': f[indices[0:8]]})
        validation.append({'g': g[indices[8:16]], 'f': f[indices[8:16]]})
        test.append({'g': g[indices[16:24]], 'f': f[indices[16:24]]})

    return train, test, validation


def load_samples(set):
    positive_samples = []
    negative_samples_GF = []
    for author in set:
        for i in range(8):
            for j in range(i+1, 8):
                positive_samples.append(np.abs(author['g'][i] - author['g'][j]))
                negative_samples_GF.append(np.abs(author['f'][i] - author['g'][j]))

    train_samples = {'positive': positive_samples, 'negative': negative_samples_GF}

    return train_samples


def load_feature_vector(author):
    f = open('saved_features_cedar55/author-%d-forgeries' % (author + 1), 'r')
    g = open('saved_features_cedar55/author-%d-genuine' % (author + 1), 'r')

    return np.load(g), np.load(f)


def create_database():
    for author in range(10, 55):
        genuine = compute_feature_vector(author + 1, 24, 'g')
        forgeries = compute_feature_vector(author + 1, 24, 'f')
        author_genuine = genuine
        author_forgeries = forgeries
        fg = open('saved_features_cedar55/author-%d-genuine' % (author+1), 'w+')
        ff = open('saved_features_cedar55/author-%d-forgeries' % (author+1), 'w+')
        np.save(fg, author_genuine)
        np.save(ff, author_forgeries)

        print 'Saved features from author %d' % (author+1)



def compute_feature_vector(author, number, type):
    features = []

    for i in range(number):
        if(type == 'g'):
            path = PATH + 'full_org/original_%d_%d.png' % (author, (i + 1))
        elif(type == 'f'):
            path = PATH + 'full_forg/forgeries_%d_%d.png' % (author, (i + 1))
        else:
            raise ValueError("type must be g or f")

        features.append(extract_feature_vector(np.array(Image.open(path).convert('L'))))

    return features
