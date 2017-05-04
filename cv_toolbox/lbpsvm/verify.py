"""Verify a signature
Usage:
    verify.py multi <dataset_folder> <total_authors> <number_enrolment> <sig_type> <negative_folder> <number_negative>
    verify.py mono <dataset_folder> <total_authors> <number_enrolment> <sig_type> <negative_folder>

"""

from lbp import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2
from docopt import docopt
from core.util import gradient
from features.shog import shog
from sklearn import preprocessing
from sklearn import svm, ensemble
from hu2016.feature import extract_feature_vector
import glob
import os.path as P
import numpy as np

def pause():
    programPause = input("Press the <ENTER> key to continue...")

def compute_feature(fn):
    image = cv2.imread(fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return extract_feature_vector(gray)


def stats(summary):
    right = 0
    for row in summary:
        right+=(int(row['predicted']) == row['label'])

    return (right, right/float(len(summary)))
def main(args):
    np.random.seed(seed=42)
# set(glob("*")) - set(glob("eph"))

    print(args)
    dataset_folder = args['<dataset_folder>']
    total_authors = int(args['<total_authors>'])
    number_enrolment = int(args['<number_enrolment>'])
    sig_type = args['<sig_type>']
    negative_folder = args['<negative_folder>']
    number_negative = int(args['<number_negative>'])
    # --test ~/playground/cv_toolbox/nn/renamed-biosecurid/u0089/g/u0089s0001_sg0002g-r.png \

    if(number_enrolment > total_authors):
        raise 'Number enrolment must be lower than total authors'

    negative_fns = []
    for ext in ('*.gif', '*.png', '*.jpg', '*v*.bmp', '*V*.BMP'):
       negative_fns.extend(glob.glob(P.join(negative_folder+'/*/', ext), recursive=True))
    negative_fns = np.array(negative_fns)

    # loop over the training images
    for user in range(1, number_enrolment):
        u = 'u%04d' % user
        
        # first signature of each ession of user 00001
        pattern_positive = '%s*1g-%s.png' % (u, sig_type)
        pattern_all = '%s*-%s.png' % (u, sig_type)
        pattern_random = '*s0001*1g*-%s.png' % sig_type
        genuine_folder = P.join(u, 'g')
        forgery_folder = P.join(u, 'f')
        positive_fns = glob.glob(P.join(P.join(dataset_folder, genuine_folder), pattern_positive))
        all_positive_fns = glob.glob(P.join(P.join(dataset_folder, genuine_folder), pattern_all))
        all_fns = glob.glob(P.join(P.join(dataset_folder, genuine_folder), pattern_all))

        test_fns_genuine = list(set(all_positive_fns) - set(positive_fns))
        test_fns_skilled_forgery = glob.glob(P.join(P.join(dataset_folder, forgery_folder), pattern_all))
        test_fns_random_forgery = []
        for random_forgery_user in range(number_enrolment+1, total_authors+1):
            ru = 'u%04d' % random_forgery_user
            genuine_folder = P.join(ru, 'g')
            test_fns_random_forgery.extend(glob.glob(P.join(P.join(dataset_folder, genuine_folder), pattern_random), recursive=True))
                    
        # initialize the local binary patterns descriptor along with
        # the data and label lists
        data = []
        labels = []

        for positive_fn in positive_fns:
            # load the image, convert it to grayscale, and describe it
            hist = compute_feature(positive_fn)
            # hist = desc.describe(gray)

            # extract the label from the image path, then update the
            # label and data lists
            labels.append(1)
            data.append(hist)

        f_negative_fns = negative_fns.copy()
        np.random.shuffle(f_negative_fns)
        f_negative_fns = f_negative_fns[:number_negative]
        # loop over the training images
        for negative_fn in f_negative_fns:
            # load the image, convert it to grayscale, and describe it
            hist = compute_feature(negative_fn)

            # extract the label from the image path, then update the
            # label and data lists
            labels.append(0)
            data.append(hist)

        # print("\n".join(positive_fns))
        # print("\n".join(f_negative_fns))
        # train a Linear SVM on the data
        model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10)
        model.fit(data, labels)
        summary = {'genuine': [], 'skilled': [], 'random': []}
        for test_fn in test_fns_genuine:
            hist = compute_feature(test_fn)

            hist = hist.reshape(1, -1)
            result = model.predict_proba(hist)
            neg, pos = result[0]
            prediction = neg <= 0.1
            summary['genuine'].append({'predicted': prediction, 'label': 1})

        for test_fn in test_fns_skilled_forgery:
            hist = compute_feature(test_fn)

            hist = hist.reshape(1, -1)
            result = model.predict_proba(hist)
            neg, pos = result[0]
            prediction = neg <= 0.1
            summary['skilled'].append({'predicted': prediction, 'label': 0})
        
        for test_fn in test_fns_random_forgery:
            hist = compute_feature(test_fn)

            hist = hist.reshape(1, -1)
            result = model.predict_proba(hist)
            neg, pos = result[0]
            prediction = neg <= 0.1
            summary['random'].append({'predicted': prediction, 'label': 0})
        
        # print(summary)
        print('[genuine]', u, stats(summary['genuine']))
        print('[skilled]', u, stats(summary['skilled']))
        print('[random]', u, stats(summary['random']))

if __name__ == "__main__":
    main(docopt(__doc__))