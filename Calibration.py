import time
from sklearn import datasets
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import fetch_mldata
from sklearn.ensemble import AdaBoostClassifier

data = datasets.load_breast_cancer()
#data = datasets.load_iris()
#data = datasets.load_digits()
#data = fetch_mldata('datasets-UCI credit-g')
#data = fetch_mldata('MNIST original')

classifiers = []

ada = AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=100, random_state=1)
classifiers.append([ada, "AdaBoost-ed tree"])

ada_cal = CalibratedClassifierCV(AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=100, random_state=1), cv=2, method='isotonic')
classifiers.append([ada_cal, "calibrated AdaBoost-ed tree (isotonic)"])

ada_cal2 = CalibratedClassifierCV(AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(),n_estimators=100, random_state=1), cv=2, method='sigmoid')
classifiers.append([ada_cal2, "calibrated AdaBoost-ed tree (sigmoid)"])

neighbors = 10;
knn = KNeighborsClassifier(n_neighbors=neighbors)
classifiers.append([knn, "kNN"])

knn_cal = CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=neighbors), cv=2, method='isotonic')
classifiers.append([knn_cal, "calibrated kNN (isotonic)"])

knn_cal2 = CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=neighbors), cv=2, method='sigmoid')
classifiers.append([knn_cal2, "calibrated kNN (sigmoid)"])


for classifier, label in classifiers:
    start = time.time()
    scores = cross_val_score(classifier, data.data, data.target, cv=10, scoring="neg_log_loss")
    stop = time.time()
    print("%20s log_loss: %0.2f (+/- %0.2f), time:%.4f" % (label, scores.mean(), scores.std() * 2, stop - start))

for classifier, label in classifiers:
    start = time.time()
    scores = cross_val_score(classifier, data.data, data.target, cv=10, scoring="neg_mean_squared_error")
    stop = time.time()
    print("%20s squared error: %0.2f (+/- %0.2f), time:%.4f" % (label, scores.mean(), scores.std() * 2, stop - start))

