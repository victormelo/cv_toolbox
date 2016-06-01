from sklearn import svm, ensemble
import cedar55
import sigcomp2011
import core
import numpy as np


def run_sigcomp2011():
    train, test = sigcomp2011.load_dataset()
    samples = sigcomp2011.load_samples(train)

    X_train, y_train = core.training_set(samples)
    forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)

    for t in test:
        for questioned in t['questioned']:
            d = np.abs(questioned - t['ref'])
            print forest.predict_proba(d).max()

def run_cedar55():

    train, test, validation = cedar55.load_dataset()
    samples = core.load_samples(train)

    X_train, y_train = core.training_set(samples)


    forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)

    print len(samples['positive']), len(samples['negative'])
    print samples['positive'][0][0], samples['negative'][0][0]

    exit()
    score = []
    label = []
    for i in range(55):
        reference = train[i]['g']
        for query in validation[i]['g']:
            d = np.abs(query - reference)
            score.append(forest.predict_proba(d)[:, 1].min())
            label.append(1)

        for query in validation[i]['f']:
            d = np.abs(query - reference)
            score.append(forest.predict_proba(d)[:, 1].min())
            label.append(0)
    fpr, tpr, _ = roc_curve(label, score)
    auc(fpr, tpr)


def main():
    run_sigcomp2011()


if __name__ == '__main__':
    main()
