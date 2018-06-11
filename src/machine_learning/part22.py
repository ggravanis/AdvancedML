#!/usr/bin/env python
# coding: utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
import time


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

    plt.scatter(x=x, y=y, s=1)
    plt.show()
    df = df.head(n)
    return df['label']


if __name__ == "__main__":
    # Set the paths
    save_path = "../../results/part22/"
    print "Loading the data..."
    # Load the data

    X_test = load_data('part1_test.csv')  # 3980

    y_test = load_data('test_labels.csv')
    print "Data loaded."
    print "creating labeled arrays"

    X = bag_of_words(X_test)

    keys = get_n_most_significant(X, 3500)
    X = X[keys]
    y = y_test['reference']
    X = np.array(X)
    y = np.array(y)
    print