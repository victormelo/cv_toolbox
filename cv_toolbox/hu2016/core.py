import numpy as np


def training_set(train_samples):
    n_positives = len(train_samples['positive'])
    n_negatives = len(train_samples['negative'])
    y_train = np.concatenate((np.ones(n_positives, dtype='uint8'), np.zeros(n_negatives, dtype='uint8')))
    X_train = np.concatenate((train_samples['positive'], train_samples['negative']))

    return X_train, y_train
