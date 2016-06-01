from PIL import Image
import numpy as np
from feature import extract_feature_vector
import os
import shutil

PATH = '/home/victor/mestrado-local/bases/sigcomp-2011/'

FOLDERS = os.listdir(PATH+'Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Questioned(1287)/')


def load_dataset():
    train = []
    test = []
    for i in range(16):
        g, f = load_feature_vector_train(i)
        train.append({'g': g, 'f': f})

    for author in FOLDERS:
        r, q = load_feature_vector_test(author)
        test.append({'ref': r, 'questioned': q})

    return train, test


def load_samples(set):
    positive_samples = []
    negative_samples_GF = []
    cleaned_set = []
    for author in set:
        if(len(author['g']) != 0 and len(author['f']) != 0):
            cleaned_set.append(author)

    for author in cleaned_set:
        for i in range(len(author['g'])):
            for j in range(i+1, len(author['g'])-1):
                positive_samples.append(np.abs(author['g'][i] - author['g'][j]))
            for forg in author['f']:
                negative_samples_GF.append(np.abs(author['g'][i] - forg))

    positive_samples = np.array(positive_samples)
    negative_samples_GF = np.array(negative_samples_GF)
    negatives = np.random.permutation(len(negative_samples_GF))

    negative_samples_GF = negative_samples_GF[negatives[:len(positive_samples)]]

    train_samples = {'positive': positive_samples, 'negative': negative_samples_GF}

    return train_samples

def load_feature_vector_test(author):
    r = open('saved_features_sigcomp2011/test/reference/author-%s' % author)
    q = open('saved_features_sigcomp2011/test/questioned/author-%s-genuine' % author)

    return np.load(r), np.load(q)


def load_feature_vector_train(author):
    f = open('saved_features_sigcomp2011/train/author-%d-forgeries' % (author + 1), 'r')
    g = open('saved_features_sigcomp2011/train/author-%d-genuine' % (author + 1), 'r')

    return np.load(g), np.load(f)

def create_database_reference():
    dirpath = 'saved_features_sigcomp2011/test/reference/'
    shutil.rmtree(dirpath)
    os.mkdir(dirpath)
    reference_path = PATH + 'Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Reference(646)/'

    for folder in os.listdir(reference_path):
        author_vector = []
        for file in os.listdir(reference_path + folder):
            author_vector.append(extract_feature_vector(np.array(Image.open(reference_path + folder + '/' + file).convert('L'))))

        fr = open(dirpath + 'author-%s' % folder, 'w+')
        np.save(fr, author_vector)

        print 'Saved features from author %s' % folder

def create_database_questioned():
    dirpath = 'saved_features_sigcomp2011/test/questioned/'
    shutil.rmtree(dirpath)
    os.mkdir(dirpath)
    questioned_path = PATH + 'Testdata_SigComp2011/SigComp11-Offlinetestset/Dutch/Questioned(1287)/'

    for folder in os.listdir(questioned_path):
        author_genuine = []
        author_forgeries = []
        for file in os.listdir(questioned_path + folder):
            # All forgeries files have length 14
            if len(file) == 14:
                author_forgeries.append(extract_feature_vector(np.array(Image.open(questioned_path + folder + '/' + file).convert('L'))))
            # All genuine files have length 10
            elif len(file) == 10:
                author_genuine.append(extract_feature_vector(np.array(Image.open(questioned_path + folder + '/' + file).convert('L'))))

        fg = open(dirpath + 'author-%s-genuine' % folder, 'w+')
        ff = open(dirpath + 'author-%s-forgeries' % folder, 'w+')
        np.save(fg, author_genuine)
        np.save(ff, author_forgeries)

        print 'Saved features from author %s' % folder



def create_database_train():
    dirpath = 'saved_features_sigcomp2011/train/'
    shutil.rmtree(dirpath)
    os.mkdir(dirpath)
    for author in range(16):
        # For both online and offline modes, signatures of 16
        # reference writers and skilled forgeries of these signatures.
        author_genuine = compute_feature_vector(author+1, 'g', 'trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine/')
        author_forgeries = compute_feature_vector(author+1, 'f', 'trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries/')
        fg = open(dirpath + 'author-%d-genuine' % (author+1), 'w+')
        ff = open(dirpath + 'author-%d-forgeries' % (author+1), 'w+')
        np.save(fg, author_genuine)
        np.save(ff, author_forgeries)

        print 'Saved features from author %d' % (author+1)


def compute_feature_vector(author, type='g', type_path='trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Genuine/'):
    # trainingSet/OfflineSignatures/Dutch/TrainingSet/Offline Forgeries/
    features = []
    import matplotlib.pyplot as plt
    path = PATH + type_path
    for file in os.listdir(path):
        # Genuine signatures are named according to the following convention
        # (the same for all data sets): III_NN.*, where III is the ID of the
        # reference writer and NN is an index of the signature, i.e., it is
        # the NNth authentic signature contributed by writer III.
        if type == 'g':
            if(file[:3] == '%03d' % author):
                features.append(extract_feature_vector(np.array(Image.open(path+file).convert('L'))))

        elif type == 'f':
            if(file[4:7] == '%03d' % author):
                features.append(extract_feature_vector(np.array(Image.open(path+file).convert('L'))))

        else:
            raise ValueError("type must be g or f")

    return features


def main():
    # create_database_train()
    # create_database_reference()
    # create_database_questioned()
    pass

if __name__ == '__main__':
    main()
