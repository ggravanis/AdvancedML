#!/usr/bin/env python
# coding: utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.metrics import precision_recall_curve, average_precision_score, hamming_loss, \
    label_ranking_average_precision_score, roc_auc_score, accuracy_score

from itertools import cycle
import random


def binary_to_text(df):
    y_labels = []
    labels_cumsum = {}
    for index, item in df.iterrows():
        labels = []
        for label_index, value in item.iteritems():
            if np.isnan(value):
                continue
            if int(value) == 1:
                labels.append(label_index)
                if labels_cumsum.get(label_index) is None:
                    labels_cumsum[label_index] = 0
                else:
                    labels_cumsum[label_index] = labels_cumsum.get(label_index) + 1
        y_labels.append(labels)
    return y_labels, labels_cumsum


def load_data(filename):
    load_path = "../../data/DeliciousMIL/"
    data = pd.read_csv(load_path + filename)
    data = data.set_index(['Unnamed: 0'])
    return data


def bag_of_words(df):
    d = {}
    for row, items in df.iterrows():
        my_dict = {}
        for item in items:
            if np.isnan(item):
                continue
            if int(item) in my_dict:
                my_dict[int(item)] += 1
            else:
                my_dict[item] = 1

        d[row] = my_dict

    df = pd.DataFrame.from_dict(d)
    df = df.transpose()
    df = pd.DataFrame(df).fillna(0)
    return df


def get_n_most_significant(df, n):
    temp_dict = {}
    for index, column in df.items():
        temp_dict[index] = int(sum(column))

    df = pd.DataFrame(temp_dict, index=[0])
    df = df.transpose()
    df = df.reset_index(drop=False)
    df = df.rename({"index": "label", 0: "value"}, axis='columns')

    # df['value'].plot.kde()
    # plt.show()

    df.label = df.label.astype(int)
    df = df.sort_values(by="value", ascending=False)
    x = []
    y = []
    label = []
    counter = 0
    for index, item in df.iterrows():
        x.append(counter)
        y.append(item['value'])
        label.append(item['label'])
        counter += 1

    # plt.scatter(x=x, y=y, s=1)
    # plt.show()
    df = df.head(n)
    return df['label']


if __name__ == "__main__":
    # Set the paths
    save_path = "../../results/part23/"
    print "Loading the data..."
    # Load the data

    X_test = load_data('part1_test.csv')  # 3980

    y_test = load_data('test_labels.csv')
    print "Data loaded."
    print "creating labeled arrays"

    X = bag_of_words(X_test)

    keys = get_n_most_significant(X, 3500)
    X = X[keys]
    y = y_test['grammar']
    # y = y_test['language']
    X = np.array(X)
    y = np.array(y)

    fig, axes = plt.subplots(ncols=2, nrows=5, sharey='all', sharex='row', figsize=(10, 22))
    fig.suptitle("Learning Curves")
    fig.text(0.5, 0.04, 'Iteration', ha='center')
    fig.text(0.04, 0.5, 'Accuracy %', va='center', rotation='vertical')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95, left=0.11, bottom=0.08)

    for k in range(0, 10):
        if k % 2 == 0:
            j = 0
        else:
            j = 1
        if k == 1:
            m = 0
        else:
            m = k / 2
        print m
        # split test set in two sets equally distributed
        X_test, X_pool, y_test, y_pool = train_test_split(X, y, test_size=0.50, random_state=7)

        X_train = []
        y_train = []
        for i in range(len(y_pool)):
            if y_pool[i] == 0:
                if len(y_train) < 4:
                    X_train.append(X_pool[i])
                    y_train.append(y_pool[i])
                    np.delete(X_pool, i)
                    np.delete(y_pool, i)
                else:
                    continue

            if y_pool[i] == 1:
                if len(y_train) < 8:
                    X_train.append(X_pool[i])
                    y_train.append(y_pool[i])
                    np.delete(X_pool, i)
                    np.delete(y_pool, i)
                else:
                    break

        # Uncertainty Sampling Initialization
        X_train_US = np.array(X_train)
        y_train_US = np.array(y_train)
        X_pool_US = np.array(X_pool)
        y_pool_US = np.array(y_pool)

        # Random Sampling Initialization
        X_train_RS = np.array(X_train)
        y_train_RS = np.array(y_train)
        X_pool_RS = np.array(X_pool)
        y_pool_RS = np.array(y_pool)

        # Initialize storage arrays

        us_acc = []
        rs_acc = []
        for iteration in range(10):
            clf_US = SVC(kernel='linear', C=1, probability=True)

            # Train classifier with Uncertainty Sampling Train set
            clf_US.fit(X_train_US, y_train_US)
            prob_US = clf_US.predict_proba(X_pool_US)
            y_pred_US = clf_US.predict(X_test)

            temp = [np.abs(item[1] - 0.5) for item in prob_US]  # we focus on minority class item[1]
            u_sample = np.argmin(temp)

            # Train classifier with Random Sampling Train set
            clf_RS = SVC(kernel='linear', C=1, probability=True)
            clf_RS.fit(X_train_RS, y_train_RS)
            prob_RS = clf_RS.predict_proba(X_pool_RS)
            y_pred_RS = clf_RS.predict(X_test)

            r_sample = random.randint(0, len(y_pool_RS))

            # Store scores in arrays
            us_acc.append(accuracy_score(y_pred=y_pred_US, y_true=y_test))
            rs_acc.append(accuracy_score(y_pred=y_pred_RS, y_true=y_test))

            print "Iteration {}".format(iteration + 1)

            print "Uncertainty Sampling AUC: ", roc_auc_score(y_score=y_pred_US, y_true=y_test)
            print "Uncertainty Sampling Accuracy: ", accuracy_score(y_pred=y_pred_US, y_true=y_test)
            print "Uncertainty Sampling Pool size", X_pool_US.shape
            print "Uncertainty Sampling Train size", X_train_US.shape
            print "The Uncertainty sample number is {} and its value is {}".format(u_sample, temp[u_sample])

            print "---------------------------------------------"

            print "Random Sampling AUC: ", roc_auc_score(y_score=y_pred_RS, y_true=y_test)
            print "Random Sampling Accuracy: ", accuracy_score(y_pred=y_pred_RS, y_true=y_test)
            print "Random Sampling Pool size", X_pool_RS.shape
            print "Random Sampling Train size", X_train_RS.shape
            print "The Random sample number is {} and its value is {}".format(r_sample, temp[r_sample])

            # Update train and pool set for Uncertainty Sampling
            X_train_US = np.vstack((X_train_US, X_pool_US[u_sample, :]))
            y_train_US = np.hstack((y_train_US, y_pool_US[u_sample]))
            X_pool_US = np.delete(X_pool_US, u_sample, 0)
            y_pool_US = np.delete(y_pool_US, u_sample, 0)

            # Update train and pool set for Random Sampling
            X_train_RS = np.vstack((X_train_RS, X_pool_RS[r_sample, :]))
            y_train_RS = np.hstack((y_train_RS, y_pool_RS[r_sample]))
            X_pool_RS = np.delete(X_pool_RS, r_sample, 0)
            y_pool_RS = np.delete(y_pool_RS, r_sample, 0)

        axes[m, j].plot(us_acc)
        axes[m, j].plot(rs_acc)

        # axes[i, 0].set_ylabel("General Iteration {}".format(k))
        axes[m, j].set_title("Test {}".format(k+1))

        plt.setp([a.get_xticklabels() for a in axes[0, :]], visible=False)
        plt.setp([a.get_yticklabels() for a in axes[:, 1]], visible=False)
        fig.legend(("Uncertainty Sampling", "Random Sampling"), loc='upper right')
        fig.savefig(save_path + 'ALearningCurves.png', bbox_inches='tight')

        # plt.figure()
        # plt.plot(us_acc, label="Uncertainty Sampling")
        # plt.plot(rs_acc, label="Random Sampling")
        # plt.legend()
        # plt.title("Learning Curves")
        # plt.ylabel("Accuracy %")
        # plt.ylim((0, 1))
        # plt.xlabel("Iteration")
        # plt.savefig(save_path + "ALearningCurves_{}.png".format(k))
        # plt.show()
