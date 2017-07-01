"""Verify a signature
Usage:
    verify.py [options] multi <dataset_folder> <total_authors> <number_enrolment> <sig_type> <negative_folder> <number_negative>
    verify.py [options] mono <dataset_folder> <total_authors> <number_enrolment> <sig_type> <negative_folder>
    verify.py [options] mixed <dataset_folder> <total_authors> <number_enrolment> <sig_type> <negative_folder> <number_negative>

Options:
  --cache-dir=<s>                directory to use for cache
  --synth-sig-type=<s>            the sig type to use as real in case of mixed set
  --mixed-mode=<s>            i, ii or iii.. refer to the paper [default: i]
  

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
import os
import numpy as np
import threading
import h5py
from sklearn import linear_model
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import shutil
from sklearn import linear_model
all_random_forgeries = []
np.random.seed(seed=42)


def file_output(filename):
    return open(filename, 'w')

def result(probas, truth):
    fpr, tpr, thresholds = roc_curve(truth, np.array(probas)[:, 1])
    fnr = 1-tpr

    arg = np.abs((fpr-fnr)).argmin()
    print('fpr: %f - fnr: %f - th: %f' % (
        fpr[arg],
        fnr[arg],
        thresholds[arg]))

class TrainerThread(threading.Thread):
    def __init__(self, name, url, q):
        threading.Thread.__init__(self)
        self.name = name
        self.out = out

    def run(self):
        while not self.q.full():
            try:
                source = urllib.request.urlopen(self.url).read()
                self.q.put({'url': self.url, 'source': source})
                logging.debug('Inserting ' + str(self.url)
                              + ' : ' + str(self.q.qsize()) + ' responses in queue')
                return
            except Exception as e:
                logging.debug('Error ' + str(self.url) + ': ' + str(e))
                return


def pause():
    programPause = input("Press the <ENTER> key to continue...")


def compute_feature(fn, cache_dir):
    image = cv2.imread(fn)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bfn = P.basename(fn).replace('.', '-')
    h5fn = P.join(cache_dir, '%s.h5' % bfn)
    error = False
    if P.exists(h5fn):
        try:
            h5f = h5py.File(h5fn,'r')
            features = h5f['features'][:]
        except:
            error = True
    
    # if not P.exists(h5fn) or error:
    #     features = extract_feature_vector(gray)
    #     dn = P.dirname(h5fn)
    #     if not P.exists(dn):
    #         os.makedirs(dn)

    #     h5f = h5py.File(h5fn, 'w')
    #     h5f.create_dataset('features', data=features)
    # h5f.close()
    try:
        return features
    except:
        print(h5fn)

def stats(summary):
    right = 0
    for row in summary:
        right+=(int(row['predicted']) == row['label'])

    return (right, right/float(len(summary)))


def predict(model, data, th=0.1):
    result = model.predict_proba(data.reshape(1, -1))
    neg, pos = result[0]
    prediction = neg <= th
    return prediction

class UserEvaluatorThread(threading.Thread):
    def __init__(self, user, model, positive_fns, negative_fns, test_fns_genuine, test_fns_skilled_forgery, test_fns_random_forgery, options, results):
        threading.Thread.__init__(self)
        self.model = model
        self.test_fns_genuine = test_fns_genuine
        self.test_fns_skilled_forgery = test_fns_skilled_forgery
        self.test_fns_random_forgery = test_fns_random_forgery
        self.summary = {'genuine': [], 'skilled': [], 'random': []}
        self.user = user
        self.positive_fns = positive_fns
        self.negative_fns = negative_fns
        self.labels = []
        self.data = []
        self.options = options
        self.results = results
    
    def run(self):
        for positive_fn in self.positive_fns:
            # load the image, convert it to grayscale, and describe it
            hist = compute_feature(positive_fn, self.options['cache_dir'])
            # hist = desc.describe(gray)

            # extract the label from the image path, then update the
            # label and data lists
            self.labels.append(1)
            self.data.append(hist)

        for negative_fn in self.negative_fns:
            # load the image, convert it to grayscale, and describe it
            hist = compute_feature(negative_fn, self.options['cache_dir'])

            # extract the label from the image path, then update the
            # label and data lists
            self.labels.append(0)
            self.data.append(hist)

        
        X = []
        ytrue = []
        for test_fn in self.test_fns_genuine:
            data = compute_feature(test_fn, self.options['cache_dir'])
            X.append(data)
            ytrue.append(1)
            
        for test_fn in self.test_fns_skilled_forgery:
            data = compute_feature(test_fn, self.options['cache_dir'])
            X.append(data)
            ytrue.append(0)
        for test_fn in self.test_fns_random_forgery:
            data = compute_feature(test_fn, self.options['cache_dir'])
            X.append(data)
            ytrue.append(0)

        self.model.fit(self.data, self.labels)
        probas = self.model.predict_proba(X)
        for i, proba in enumerate(probas):
            self.results[self.user]['probas'].append(proba)
            self.results[self.user]['truth'].append(ytrue[i])
        
        # result(self.results[self.user]['probas'], self.results[self.user]['truth'])
        # self.summary['random'].append({'predicted': prediction, 'label': 0})
        # prediction = predict(self.model, data)
        # self.summary['skilled'].append({'predicted': prediction, 'label': 0})
        # prediction = predict(self.model, data)
        # self.summary['genuine'].append({'predicted': prediction, 'label': 1})
        # prediction = self.model.predict_proba(data)
        # import pdb; pdb.set_trace()
        # print('[%s]' % self.user)
        # print('[genuine]', stats(self.summary['genuine']))
        # print('[skilled]', stats(self.summary['skilled']))
        # print('[random]', stats(self.summary['random']))

def flatten_results(results):
        probas = []
        truth = []

        for key in results.keys():
            probas += results[key]['probas']
            truth += results[key]['truth']

        return probas, truth

def sanity_copy(foldername, sanity_path, fns):
    path = P.join(sanity_path, foldername)
    if(P.exists(path)):
        shutil.rmtree(path)
        
    os.makedirs(path)
    
    for fn in fns:
        shutil.copy2(fn, path)


def sanity_check(u, positive_fns, f_negative_fns, test_fns_genuine, test_fns_random_forgery, test_fns_skilled_forgery):
    path = '/home/vkslm/playground/sanity-check'

    sanity_path = P.join(path, u)
    sanity_copy('positive', sanity_path, positive_fns)
    sanity_copy('f_negative', sanity_path, f_negative_fns)
    sanity_copy('test_genuine', sanity_path, test_fns_genuine)
    sanity_copy('test_rforgery', sanity_path, test_fns_random_forgery)
    sanity_copy('test_sforgery', sanity_path, test_fns_skilled_forgery)

def evaluate(number_enrolment, sig_type, dataset_folder, total_authors, negative_fns, number_negative, options, args):
    results = {}

    for user in range(1, number_enrolment+1):
        u = 'u%04d' % user
        results[u] = {'probas': [], 'truth': []}
        pattern_all = '%s*-%s.png' % (u, sig_type)
        pattern_random = '*s0001*1g*-%s.png' % sig_type
        genuine_folder = P.join(u, 'g')
        forgery_folder = P.join(u, 'f')
        pattern_to_exclude = None
        if args['multi']:
            # first signature of each ession of user 00001
            pattern_positive = ['%s*1g-%s.png' % (u, sig_type)]
        elif args['mixed']:
            # i) 4 real samples belonging to the
            # first acquisition session (as in experiment 1 - mono
            # session), ii)  and iii) 4 real samples belonging
            # to the first session plus 4 synthetic samples belonging
            # to the second session.
            if options['mixed_mode'] == 'i':
                 # 4 real samples belonging to the first acquisition session
                pattern_positive = ['%ss0001*-%s.png' % (u, sig_type)]

            elif options['mixed_mode'] == 'ii':
                # 8 real samples belonging to the first and the second sessions,
                pattern_positive = ['%ss0001*-%s.png' % (u, sig_type),
                                    '%ss0002*-%s.png' % (u, sig_type)]

            elif options['mixed_mode'] == 'iii':
                # 4 real samples belonging
                # to the first session plus 4 synthetic samples belonging
                # to the second session.
                pattern_positive = ['%ss0001*-%s.png' % (u, sig_type),
                                    '%ss0002*-%s.png' % (u, options['synth_sig_type'])]
            
            pattern_to_exclude = ['%ss0001*-%s.png' % (u, sig_type), '%ss0002*-%s.png' % (u, sig_type)]

        positive_fns = []
        for pattern in pattern_positive:
           positive_fns.extend(glob.glob(P.join(P.join(dataset_folder, genuine_folder), pattern)))

        all_positive_fns = glob.glob(P.join(P.join(dataset_folder, genuine_folder), pattern_all))

        if pattern_to_exclude:
            to_exclude_fns = []
            for pattern in pattern_to_exclude:
               to_exclude_fns.extend(glob.glob(P.join(P.join(dataset_folder, genuine_folder), pattern)))            
            test_fns_genuine = list(set(all_positive_fns) - set(to_exclude_fns))
        else:
            test_fns_genuine = list(set(all_positive_fns) - set(positive_fns))

        all_fns = glob.glob(P.join(P.join(dataset_folder, genuine_folder), pattern_all))
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

        f_negative_fns = negative_fns.copy()
        np.random.seed(seed=42)
        np.random.shuffle(f_negative_fns)
        f_negative_fns = f_negative_fns[:number_negative]

        sanity_check(u, positive_fns, f_negative_fns, test_fns_genuine, test_fns_random_forgery, test_fns_skilled_forgery)
        # print("\n".join(positive_fns))
        # print("\n".join(f_negative_fns))
        # train a Linear SVM on the data
        model = ensemble.RandomForestClassifier(n_estimators=30, max_depth=3)

        evaluator = UserEvaluatorThread(u,
            model, 
            positive_fns, 
            f_negative_fns, 
            test_fns_genuine, 
            test_fns_skilled_forgery, 
            test_fns_random_forgery,
            options,
            results)
        evaluator.run()

        print(len(results), end='\r')

        # summary = {'genuine': [], 'skilled': [], 'random': []}
        # for test_fn in test_fns_genuine:
        #     prediction = predict(model, test_fn)
        #     summary['genuine'].append({'predicted': prediction, 'label': 1})

        # for test_fn in test_fns_skilled_forgery:
        #     prediction = predict(model, test_fn, 0)
        #     summary['skilled'].append({'predicted': prediction, 'label': 0})

        # for test_fn in test_fns_random_forgery:
        #     prediction = predict(model, test_fn, 0)
        #     summary['random'].append({'predicted': prediction, 'label': 0})

        # print(summary)
        # print('User %s' % u)
        # print('[genuine]', stats(summary['genuine']))
        # print('[skilled]', stats(summary['skilled']))
        # print('[random]', stats(summary['random']))
    # assert len(results['probas']) == (90 * 66)

    probas, truth = flatten_results(results)
    result(probas, truth)
    

def main(args):
# set(glob("*")) - set(glob("eph"))

    print(args)
    dataset_folder = args['<dataset_folder>']
    total_authors = int(args['<total_authors>']) if args['<total_authors>'] else 0
    number_enrolment = int(args['<number_enrolment>']) if args['<number_enrolment>'] else 0
    sig_type = args['<sig_type>']
    cache_dir = args['--cache-dir']
    synth_sig_type = args['--synth-sig-type']
    mixed_mode = args['--mixed-mode']
    negative_folder = args['<negative_folder>']
    number_negative = int(args['<number_negative>']) if args['<number_negative>'] else 0
    options = {
        'cache_dir': cache_dir,
        'synth_sig_type': synth_sig_type,
        'mixed_mode': mixed_mode

    }
    # --test ~/playground/cv_toolbox/nn/renamed-biosecurid/u0089/g/u0089s0001_sg0002g-r.png \

    if(number_enrolment > total_authors):
        raise 'Number enrolment must be lower than total authors'

    negative_fns = []
    for ext in ('*.gif', '*.png', '*.jpg', '*v*.bmp', '*V*.BMP'):
       negative_fns.extend(glob.glob(P.join(negative_folder+'/*/', ext), recursive=True))
    negative_fns = np.array(negative_fns)

    # loop over the training images
    evaluate(number_enrolment, sig_type, dataset_folder, total_authors, negative_fns, number_negative, options, args)


if __name__ == "__main__":
    main(docopt(__doc__))