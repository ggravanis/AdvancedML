# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 14:22:06 2017

@author: ggrav
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, zero_one_loss
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

start_time = time()

# Load the data
load_path = '../../data/part1/'
save_path = '../../results/part1/'


def learning_curve(min_val, max_val, step, lc_seed, lc_model, data_X, data_y):
    train_error = []
    test_error = []
    batches = []

    for i in range(min_val, max_val, step):
        test_size = (100 - i) / 100.0
        batches.append(1 - test_size)
        X_train_lc, X_test_lc, y_train_lc, y_test_lc = train_test_split(data_X, data_y, random_state=lc_seed,
                                                                        test_size=test_size)
        lc_model.fit(X_train_lc, y_train_lc)
        y_pred_train = lc_model.predict(X_train_lc)
        y_pred_test = lc_model.predict(X_test_lc)
        train_error.append(zero_one_loss(y_pred=y_pred_train, y_true=y_train_lc))
        test_error.append(zero_one_loss(y_pred=y_pred_test, y_true=y_test_lc))

    scores = cross_val_score(lc_model, data_X, data_y, scoring="neg_mean_squared_error", cv=10)
    return batches, train_error, test_error, scores


if __name__ == "__main__":

    seed = 100
    folds = 10
    colors = ['#0ffaa1', '#D9BDAC', '#BA6359', '#376273']
    i = 0
    j = 0

    # Ensemble Methods Parameters
    adaBoost_params = {'AdaBoost__base_estimator__max_depth': (1, 3, 5),
                       'AdaBoost__base_estimator__min_samples_leaf': (1, 3, 5),
                       'AdaBoost__base_estimator__criterion': ['entropy'],
                       'AdaBoost__base_estimator__splitter': ['random', 'best'],
                       'AdaBoost__base_estimator__max_features': ['sqrt', 'log2'],
                       'AdaBoost__n_estimators': (80, 100, 120),
                       'AdaBoost__learning_rate': (0.1, 0.3, 0.5),
                       'AdaBoost__algorithm': ['SAMME', 'SAMME.R'],
                       'AdaBoost__random_state': [seed]}

    bagging_params = {'Bagging__base_estimator__max_depth': (1, 3, 5),
                      'Bagging__base_estimator__min_samples_leaf': (1, 3, 5),
                      'Bagging__base_estimator__criterion': ['entropy'],
                      'Bagging__base_estimator__splitter': ['random', 'best'],
                      'Bagging__base_estimator__max_features': ['sqrt', 'log2'],
                      'Bagging__n_estimators': (5, 10, 15),
                      'Bagging__max_samples': (0.7, 0.9),
                      'Bagging__bootstrap': [True, False],
                      'Bagging__bootstrap_features': [True, False],
                      'Bagging__random_state': [seed],
                      'Bagging__verbose': [0]}

    gradientBoost_params = {'GradientBoost__learning_rate': (0.1, 0.3, 0.5),
                            'GradientBoost__n_estimators': (50, 100, 150)
                            }

    randomForest_params = {'RandomForest__max_depth': (1, 3, 5, 10),
                           'RandomForest__n_estimators': (50, 100, 150)
                           }
    ensembles = []

    ensembles.append(('AdaBoostDT', Pipeline([('Normalizer', Normalizer()),
                                              ('AdaBoost',
                                               AdaBoostClassifier(base_estimator=DecisionTreeClassifier()))]),
                      adaBoost_params))
    ensembles.append(('BaggingDT', Pipeline([('Normalizer', Normalizer()),
                                             ('Bagging', BaggingClassifier(base_estimator=DecisionTreeClassifier()))]),
                      bagging_params))

    ensembles.append(('GradientBoost', Pipeline([('Normalizer', Normalizer()),
                                                 ('GradientBoost', GradientBoostingClassifier())]),
                      gradientBoost_params))

    ensembles.append(('RandomForest', Pipeline([('Normalizer', Normalizer()),
                                                ('RandomForest', RandomForestClassifier())]),
                      randomForest_params))

    # Learning curves Init Figure
    fig, axes = plt.subplots(ncols=4, nrows=10, sharey='all', sharex='row', figsize=(12, 26))
    fig.suptitle("Learning Curves")
    fig.text(0.5, 0.04, 'Train set %', ha='center')
    fig.text(0.04, 0.5, 'loss %', va='center', rotation='vertical')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, left=0.11, bottom=0.08)

    accuracy_results = ['AdaBoost', 'Bagging', 'GradientBoost', 'RandomForest']
    f1_score_results = ['AdaBoost', 'Bagging', 'GradientBoost', 'RandomForest']

    for d in range(1, 11):
        d = str(d)
        dataset = pd.read_csv(load_path + d + '.csv', header=1)
        dataset = dataset.iloc[:, 1:]
        dataset = np.array(dataset)
        X = dataset[:, :-1]
        y = dataset[:, -1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.3)

        ensemble_results = []
        ensemble_names = []
        ensemble_best_results = []

        accuracy_temp_results = []
        f1_score_temp_results = []

        j = 0

        for name, model, parameters in ensembles:

            tempList = []
            grid_search = GridSearchCV(model, param_grid=parameters, cv=folds, verbose=1, return_train_score=True,
                                       n_jobs=5)
            print("Performing grid search...")
            t0 = time()
            grid_search.fit(X_train, y_train)
            df = pd.DataFrame.from_dict(grid_search.cv_results_)
            df.to_csv(save_path + name + "_" + d + "_CV_results.csv")

            y_pred = grid_search.predict(X_test)
            accuracy_temp_results.append(accuracy_score(y_test, y_pred))
            f1_score_temp_results.append(f1_score(y_test, y_pred))

            print("done in %0.3fs" % (time() - t0))
            print()
            print ("###########################################")
            print("Accuracy at test set: %0.3f" % accuracy_score(y_test, y_pred))
            print ("F score at test set: %0.3f" % f1_score(y_test, y_pred))

            print classification_report(y_test, y_pred, digits=3)
            print("Best parameters set:")
            best_parameters = grid_search.best_estimator_.get_params()
            for fold in range(folds):
                tempName = 'split' + str(fold) + '_test_score'
                tempArray = grid_search.cv_results_[tempName]
                for item in tempArray:
                    tempList.append(item)

            for param_name in sorted(parameters.keys()):
                print("\t%s: %r" % (param_name, best_parameters[param_name]))
            ensemble_names.append(name)
            ensemble_results.append(tempList)
            ensemble_best_results.append(accuracy_score(y_test, y_pred))

            batches, train_error, test_error, scores = learning_curve(min_val=10, max_val=90, step=3, lc_seed=seed,
                                                                      lc_model=grid_search.best_estimator_,
                                                                      data_X=X_train,
                                                                      data_y=y_train)

            axes[i, j].plot(batches, train_error)
            axes[i, j].plot(batches, test_error)

            axes[i, 0].set_ylabel("dataset {}".format(d))
            axes[0, j].set_title("{}".format(name))

            plt.setp([a.get_xticklabels() for a in axes[0, :]], visible=False)
            plt.setp([a.get_yticklabels() for a in axes[:, 1]], visible=False)
            fig.legend(('Train', 'Test'), loc='upper right')
            fig.savefig(save_path + 'learning_curves.png', bbox_inches='tight')

            j = j + 1
        i = i + 1

        accuracy_results = np.column_stack((accuracy_results, accuracy_temp_results))
        print accuracy_results
        f1_score_results = np.column_stack((f1_score_results, f1_score_temp_results))

        fig1, axes1 = plt.subplots(ncols=2, sharey='all', figsize=(10, 5))
        tp = fig1.suptitle("Algorithm Benchmarking, Dataset {} \n Nested {}-fold CV ".format(d, folds))
        ax = axes1[0]
        ax.set_title("CV results")
        ax.set_ylabel("Accuracy %")
        ax.set_ylim([0.5, 1])
        ax.boxplot(ensemble_results, widths=0.6)
        ax.set_xticklabels(ensemble_names, rotation=30)
        ax = axes1[1]
        width = 0.8
        ind = np.arange(len(ensemble_best_results))
        ax.set_title('Accuracy @ Test set')
        ax.bar(ind, ensemble_best_results, width, color=colors)
        ax.set_xticks(ind - width / 10)
        ax.set_xticklabels(ensemble_names, rotation=30)
        fig1.subplots_adjust(left=0.11, bottom=0.2)
        fig1.savefig(save_path + d + '.png')

    np.savetxt(save_path + "accuracy_benchmarking.csv", accuracy_results, fmt='%s', delimiter=',')
    np.savetxt(save_path + "F1_score_benchmarking.csv", f1_score_results, fmt='%s', delimiter=',')

print "Total time needed: ", time() - start_time
