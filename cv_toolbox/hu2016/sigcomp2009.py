import shutil
import os
from feature import extract_feature_vector
import numpy as np
from PIL import Image
import resource
import gc

PATH = '/home/victor/mestrado-local/bases/sigcomp 2009/SigComp2009-training/NISDCC-offline-all-001-051-6g/'

def create_database_train():
    dirpath = 'saved_features_sigcomp2009/train/'
    shutil.rmtree(dirpath)
    os.mkdir(dirpath)

    for author in range(0, 51):
        filename_list_genuine, filename_list_forgeries = filename_for(author+1)
        print filename_list_genuine
        author_genuine = compute_feature_vector(filename_list_genuine)
        author_forgeries = compute_feature_vector(filename_list_forgeries)
        fg = open(dirpath + 'author-%d-genuine' % (author+1), 'w+')
        ff = open(dirpath + 'author-%d-forgeries' % (author+1), 'w+')
        np.save(fg, author_genuine)
        np.save(ff, author_forgeries)

        print 'Saved features from author %d' % (author+1)

def compute_feature_vector(files):
    features = []
    for file in files:
        img = np.array(Image.open(PATH+file).convert('L'))
        features.append(extract_feature_vector(img))
        print 'Saved features from %s' % file
        gc.collect()

    return features

def filename_for(author):
    genuines = list()
    forgeries = list()
    for file in os.listdir(PATH):
        if( (file[7:10] == '%03d' % author) and (file[11:14] == '%03d' % author) ):
            genuines.append(file)
        elif( (file[7:10] != '%03d' % author) and (file[11:14] == '%03d' % author) ):
            forgeries.append(file)

    return genuines, forgeries

def load_dataset():
    train = []
    for i in range(11):
        g, f = load_feature_vector_train(i)
        train.append({'g': g, 'f': f})

    return train

def load_feature_vector_train(author):
    f = open('saved_features_sigcomp2009/train/author-%d-forgeries' % (author + 1), 'r')
    g = open('saved_features_sigcomp2009/train/author-%d-genuine' % (author + 1), 'r')

    return np.load(g), np.load(f)

def main():
    create_database_train()

if __name__ == '__main__':
    main()
