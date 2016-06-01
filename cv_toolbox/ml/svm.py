import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets, cross_validation, ensemble

# import some data to play with
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=0)

print X_train.shape, y_train.shape

forest = ensemble.RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train, y_train)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True).fit(X_train, y_train)


# print (forest.predict(X_test) == y_test).sum()/float(X_test.shape[0])
# print (rbf_svc.predict(X_test) == y_test).sum()/float(X_test.shape[0])
# print forest.predict_proba(X_test)
print rbf_svc.predict_proba(X_test)

