from sklearn import svm, ensemble
import cedar55
import sigcomp2011
import sigcomp2009
import core
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def run_sigcomp2011(aditional_training_data=None):
    train, test = sigcomp2011.load_dataset('dutch')
    if(aditional_training_data):
        for data in aditional_training_data:
            train.append(data)
    samples = sigcomp2011.load_samples(train)
    X_train, y_train = core.training_set(samples)

    forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)
    print 'Classifier trained'
    score = []
    label = []
    for t in test:

        for index, questioned in enumerate(t['questioned']):
            d = np.abs(questioned - t['ref'])
            score.append(forest.predict_proba(d)[:, 1].mean())
            label.append(t['labels'][index])

    evaluate_roc(score, label, plot_roc=True)

def run_cedar55():

    train, test, validation = cedar55.load_dataset()
    samples = cedar55.load_samples(train)

    X_train, y_train = core.training_set(samples)

    forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)
    print 'Classifier trained'

    # print len(samples['positive']), len(samples['negative'])
    # print samples['positive'][0][0], samples['negative'][0][0]

    score_valid = []
    label_valid = []
    for i in range(55):
        reference = train[i]['g']
        for query in validation[1]['g']:
            d = np.abs(query - reference)
            score_valid.append(forest.predict_proba(d)[:, 1].max())
            label_valid.append(1)

        for query in validation[i]['f']:
            d = np.abs(query - reference)
            score_valid.append(forest.predict_proba(d)[:, 1].max())
            label_valid.append(0)
    print 'Validation'
    threshold = evaluate_roc(score_valid, label_valid, plot_roc=True)

    score_test = []
    label_test = []
    for i in range(55):
        reference = train[i]['g']
        for query in test[i]['g']:
            d = np.abs(query - reference)
            score_test.append(forest.predict_proba(d)[:, 1].max())
            label_test.append(1)

        for query in test[i]['f']:
            d = np.abs(query - reference)
            score_test.append(forest.predict_proba(d)[:, 1].max())
            label_test.append(0)

    label_result = score_test >= threshold
    num_negatives = 0
    num_positives = 0
    fp = 0
    fn = 0
    for i in range(len(label_test)):
        if(label_test[i] == 0):
            num_negatives += 1
            if(label_test[i] != label_result[i]):
                fn += 1
        elif(label_test[i] == 1):
            num_positives += 1
            if(label_test[i] != label_result[i]):
                fp += 1
    print 'Test'
    print 'FRR', fp/float(num_positives), 'FAR', fn/float(num_negatives)
    evaluate_roc(score_test, label_test, plot_roc=True)


def evaluate_roc(score, label, plot_roc=False):

    fpr, tpr, t = roc_curve(label, score, pos_label=1)
    roc_EER = []
    cords = zip(fpr, tpr)

    dif = np.abs(1 - fpr - tpr)

    arg = np.argmin(dif)
    if (plot_roc):
        print 'auc', auc(fpr, tpr), 'threshold: ', t[arg], 'fpr', fpr[arg], 'tpr', tpr[arg]
        # plt.plot(fpr, tpr)
        # plt.show()

    return t[arg]

def main():
    # run_cedar55()
    sigcomp2009_train_data = sigcomp2009.load_dataset()
    run_sigcomp2011()


if __name__ == '__main__':
    main()
