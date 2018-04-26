import pandas as pd
import numpy as np
from collections import Counter

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import time

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.ensemble import EasyEnsemble

load_path = '../../data/part3/'
save_path = '../../results/part3/'

dataset = pd.read_csv(load_path + 'creditcard.csv', header=1)
dataset = dataset.iloc[:, 1:]
dataset = np.array(dataset)

X = dataset[:, :-1]
y = dataset[:, -1]

seed = 100

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.33)

# Data Normalization
norm_X = Normalizer()
X_train = norm_X.fit_transform(X_train)
X_test = norm_X.transform(X_test)

print "Normalization done"

def printing(y_test, y_pred, y_res):
    print "######################"
    print('Original dataset shape {}'.format(Counter(y_train)))
    try:
        print('Resampled dataset shape {}'.format(Counter(y_res)))
    except:
        print

    print "Accuracy score at Test set: ", accuracy_score(y_true=y_test, y_pred=y_pred)
    print "AUC score at Test set: ", roc_auc_score(y_true=y_test, y_score=y_pred)
    print
    print classification_report(y_test, y_pred, digits=3)


if __name__ == "__main__":

    classifiers = [('LinearSVM', SVC(C=1, kernel='linear')), ('GaussianNB', GaussianNB()),
                   ('RandomForest', RandomForestClassifier())]

    scores = {}

    for name, clf in classifiers:
        temp_dict = {}

        print "Results of {} without balancing".format(name)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        printing(y_pred=y_pred, y_test=y_test, y_res=False)
        temp_dict['Plain Method'] = roc_auc_score(y_true=y_test, y_score=y_pred)

        # Apply regular SMOTE

        sm = [('SMOTE_regular', SMOTE(kind='regular', n_jobs=-1)),
              ('SMOTE_borderline1', SMOTE(kind='borderline1', n_jobs=-1)),
              ('SMOTE_borderline2', SMOTE(kind='borderline2', n_jobs=-1)),
              ('SMOTE_svm', SMOTE(kind='svm', n_jobs=-1))]

        for sm_name, method in sm:
            X_res, y_res = method.fit_sample(X_train, y_train)
            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_test)
            printing(y_pred=y_pred, y_test=y_test, y_res=y_res)
            temp_dict[sm_name] = roc_auc_score(y_true=y_test, y_score=y_pred)

        # NearMiss
        nm = [('NearMiss1', NearMiss(random_state=seed, n_jobs=-1, version=1)),
              ('NearMiss2', NearMiss(random_state=seed, n_jobs=-1, version=2)),
              ('NearMiss3', NearMiss(random_state=seed, n_jobs=-1, version=3))]

        for nm_name, method in nm:
            X_res, y_res = method.fit_sample(X_train, y_train)
            clf.fit(X_res, y_res)
            y_pred = clf.predict(X_test)
            printing(y_pred=y_pred, y_test=y_test, y_res=y_res)
            temp_dict[nm_name] = roc_auc_score(y_true=y_test, y_score=y_pred)

        # Easy Ensemble
        ee = EasyEnsemble(random_state=seed)
        X_res, y_res = ee.fit_sample(X_train, y_train)
        mean_auc = []
        for i in range(len(y_res)):
            clf.fit(X_res[i], y_res[i])
            y_pred = clf.predict(X_test)
            printing(y_pred=y_pred, y_test=y_test, y_res=y_res[i])
            mean_auc.append(roc_auc_score(y_true=y_test, y_score=y_pred))

        temp_dict['EasyEnsemble'] = float(sum(mean_auc)) / len(mean_auc)

        print "Easy Ensemble 10-folds mean auc", float(sum(mean_auc)) / len(mean_auc)
        scores[name] = temp_dict

    df = pd.DataFrame(scores)
    df.to_csv(save_path + "scores.csv")

    print df

