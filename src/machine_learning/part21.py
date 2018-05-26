#!/usr/bin/env python
# coding: utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.model_selection import train_test_split

# Set the paths
load_path = "../../data/DeliciousMIL/"
save_path = "../../results/part21/"

# Load the data
X_test = pd.read_csv(load_path + 'part1_test.csv', na_values=0, keep_default_na=True)
X_test = pd.DataFrame(X_test).fillna(0)
X_test = X_test.set_index(['Unnamed: 0'])
X_test = X_test.iloc[:, 1:]
X_test = np.array(X_test)

X_train = pd.read_csv(load_path + 'part1_train.csv', na_values=0, keep_default_na=True)
X_train = pd.DataFrame(X_train).fillna(0)
X_train = X_train.set_index(['Unnamed: 0'])
X_train = X_train.iloc[:, 1:]
X_train = np.array(X_train)

y_train = pd.read_csv(load_path + 'train_labels.csv', na_values=0, keep_default_na=True)
y_train = pd.DataFrame(y_train).fillna(0)
y_train = y_train.iloc[:, 1:]
# y_train = np.array(y_train)

y_test = pd.read_csv(load_path + 'test_labels.csv', na_values=0, keep_default_na=True)
y_test = pd.DataFrame(y_test).fillna(0)
y_test = y_test.iloc[:, 1:]


def binary_to_text(df):
    y_labels = []
    for index, item in df.iterrows():
        labels = []
        for label_index, value in item.iteritems():
            if int(value) == 1:
                labels.append(label_index)
        y_labels.append(labels)
    return y_labels


# y_test = np.array(y_test)

y_test = binary_to_text(y_test)
y_train = binary_to_text(y_train)

if __name__ == "__main__":

    clf = OneVsRestClassifier(estimator=SVC(kernel='linear', C=500, probability=True, verbose=False), n_jobs=-1)

    # X_train, X_test, y_train, y_test = train_test_split(X_test, y_test, random_state=7, test_size=0.3)
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    my_metrics = metrics.classification_report(y_test, predictions)
    print my_metrics
