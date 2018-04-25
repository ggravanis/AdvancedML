#!/usr/bin.env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from time import time
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Normalizer
from sklearn.ensemble import BaggingClassifier

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB,GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



#Load the data

path = '../../data/'
dataset = pd.read_csv(path + '1.csv', header=1)
data = dataset.iloc[:, 1:]
data = np.array(data)


X = data[:, :-1]
y = data[:, -1]


#Define the pipeline
pipe1 = Pipeline([('clf1', DecisionTreeClassifier()),('bagg1', BaggingClassifier())])
#pipe2 = Pipeline([('clf2', MultinomialNB()), ('bagg2', BaggingClassifier())])

# 'clf2__kernel': ['linear'],
# 'clf2__C': (1, 10, 100, 1000),

#Define the parameters of Grid Search
param1 = {  'clf1__max_depth': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            'clf1__min_samples_leaf': (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            'clf1__criterion': ['gini', 'entropy'],
            'clf1__splitter': ['random', 'best'],
            'clf1__max_features': ['auto', 'sqrt', 'log2'],
            'bagg1__n_estimators': (10, 20, 30, 40, 50, 60, 70),
            # 'bagg1__max_features': [10],
            }
# param2 = {
#             'bagg2__n_estimators': (10, 20, 30, 40, 50, 60, 70),
#             'bagg2__max_features' : [0.7, 0.8, 0.9, 1.0],
#              }


pipes = [(pipe1, param1)]# ,(pipe2,param2)]

if __name__ == "__main__":

    for pipeline, parameters in pipes:

        grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, n_jobs=-1, verbose=1, cv=7)

        print("Performing grid search...")
        print("pipeline:", [name for name, _ in pipeline.steps])
        print("parameters:")
        print(parameters)
        t0 = time()
        grid_search.fit(X, y)
        print("done in %0.3fs" % (time() - t0))
        print()

        print("Best score: %0.3f" % grid_search.best_score_)
        print("Best parameters set:")
        best_parameters = grid_search.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))