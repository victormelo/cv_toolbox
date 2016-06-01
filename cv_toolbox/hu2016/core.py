import numpy as np


def training_set(train_samples):
    n_positives = len(train_samples['positive'])
    n_negatives = len(train_samples['negative'])
    y_train = np.concatenate((np.ones(n_positives, dtype='uint8'), np.zeros(n_negatives, dtype='uint8')))
    X_train = np.concatenate((train_samples['positive'], train_samples['negative']))

    return X_train, y_train


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
