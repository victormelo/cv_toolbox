"""Verify a signature
Usage:
    verify.py (--test <test>)... (--positive <positive>)... (--negative <negative>)...
    verify.py <dataset_folder> <user> <sig_type> <negative_folder>

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

def compute_feature(fn):
    image = cv2.imread(fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return extract_feature_vector(gray)

def main(args):
    print(args)

    test_fns = args['<test>']
    positive_fns = args['<positive>']
    negative_fns = args['<negative>']

    # initialize the local binary patterns descriptor along with
    # the data and label lists
    desc = LocalBinaryPatterns(24, 3)
    data = []
    labels = []

    # loop over the training images
    for positive_fn in positive_fns:
        # load the image, convert it to grayscale, and describe it
        hist = compute_feature(positive_fn)
        # hist = desc.describe(gray)

        # extract the label from the image path, then update the
        # label and data lists
        labels.append(1)
        data.append(hist)

    # loop over the training images
    for negative_fn in negative_fns:
        # load the image, convert it to grayscale, and describe it
        hist = compute_feature(negative_fn)

        # extract the label from the image path, then update the
        # label and data lists
        labels.append(0)
        data.append(hist)

    # train a Linear SVM on the data
    model = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(data, labels)
    # model = LinearSVC(C=30.0, random_state=42)

    for test_fn in test_fns:
        hist = compute_feature(test_fn)

        hist = hist.reshape(1, -1)
        result = model.predict_proba(hist)
        print(result, test_fn)


if __name__ == "__main__":
    main(docopt(__doc__))