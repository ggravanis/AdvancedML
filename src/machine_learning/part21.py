#!/usr/bin/env python
# coding: utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import time
from sklearn.metrics import precision_recall_curve, average_precision_score, hamming_loss, \
    label_ranking_average_precision_score

from itertools import cycle


def binary_to_text(df):
    y_labels = []
    labels_cumsum = {}
    for index, item in df.iterrows():
        labels = []
        for label_index, value in item.iteritems():
            if int(value) == 1:
                labels.append(label_index)
                if labels_cumsum.get(label_index) is None:
                    labels_cumsum[label_index] = 0
                else:
                    labels_cumsum[label_index] = labels_cumsum.get(label_index) + 1
        y_labels.append(labels)
    return y_labels, labels_cumsum


def load_data(filename, size):
    load_path = "../../data/DeliciousMIL/"
    data = pd.read_csv(load_path + filename, na_values=0, keep_default_na=True)
    data = pd.DataFrame(data).fillna(0)
    data = data.set_index(['Unnamed: 0'])
    if size > 0:
        data = data.iloc[:size, :]
    else:
        data = data.iloc[:, :]
    return data


if __name__ == "__main__":

    # Set the paths

    save_path = "../../results/part21/"

    # Load the data
    X_train = load_data('part1_train.csv', 150)
    X_test = load_data('part1_test.csv', 150)
    y_train = load_data('train_labels.csv', 150)
    y_test = load_data('test_labels.csv', 150)

    # create arrays with labels
    y_test, test_distribution = binary_to_text(y_test)
    y_train, train_distribution = binary_to_text(y_train)

    index = ['programming', 'style', 'reference', 'java', 'web', 'internet', 'culture', 'design', 'education',
             'language', 'books', 'writing', 'computer', 'english', 'politics', 'history', 'philosophy', 'science',
             'religion', 'grammar']

    labels_test = pd.Series(test_distribution, index=index)
    labels_train = pd.Series(train_distribution, index=index)

    # Plot label distribution barchart
    plt.figure(figsize=(12, 10))
    ind = np.arange(1, len(index) + 1)
    p1 = plt.bar(ind, labels_train.values)
    p2 = plt.bar(ind, labels_test.values, bottom=labels_train.values)
    plt.xticks(ind, labels_test.keys(), rotation='vertical')
    plt.legend((p1[0], p2[0]), ('Train', 'Test'))
    plt.title("Label distribution \n DeliciousMIL dataset")
    plt.savefig(save_path + 'label_distribution.png')

    # Binarize the output vector
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    y_test = mlb.transform(y_test)

    # Define and train the multi-label classifier
    estimators = [('svm_linear', SVC(kernel='linear')), ("NB",GaussianNB()), ("DT",DecisionTreeClassifier())]
    for name,estimator in estimators:

        clf = OneVsRestClassifier(estimator=estimator, n_jobs=5)

        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)

        my_metrics = metrics.classification_report(y_test, predictions)
        print my_metrics
        with open(save_path + "{} classification_report.txt".format(name), "w") as text_file:
            text_file.write(my_metrics)

        print "Hamming loss", hamming_loss(y_true=y_test, y_pred=predictions)
        print "Ranking", label_ranking_average_precision_score(y_true=y_test, y_score=predictions)

        # For each label
        precision = {}
        recall = {}
        average_precision = {}
        for i in range(len(y_test[0])):
            precision[index[i]], recall[index[i]], _ = precision_recall_curve(y_test[:, i], predictions[:, i])
            average_precision[index[i]] = average_precision_score(y_test[:, i], predictions[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), predictions.ravel())
        average_precision["micro"] = average_precision_score(y_test, predictions, average="micro")
        print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))

        plt.figure()
        plt.step(recall['micro'], precision['micro'], color='b', alpha=0.2, where='post')
        plt.fill_between(recall["micro"], precision["micro"], step='post', alpha=0.2, color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(
            'Average precision score, micro-averaged over all labels: AP={0:0.2f} \n estimator: {0}'.format(average_precision["micro"],name))
        plt.savefig(save_path + '{}_average_precision.png'.format(name))
        # plt.show()

        # setup plot details
        colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])

        plt.figure(figsize=(10, 12))
        f_scores = np.linspace(0.2, 0.8, num=4)
        lines = []
        labels = []
        for f_score in f_scores:
            x = np.linspace(0.01, 1)
            y = f_score * x / (2 * x - f_score)
            l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
            plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

        lines.append(l)
        labels.append('iso-f1 curves')
        l, = plt.plot(recall["micro"], precision["micro"], color='black', lw=2.5)
        lines.append(l)
        labels.append('micro-average area = {0:0.2f}'
                      ''.format(average_precision["micro"]))

        for i, color in zip(range(len(y_test[0])), colors):
            l, = plt.plot(recall[index[i]], precision[index[i]], color=color, lw=1)
            lines.append(l)
            labels.append('{0} , area = {1:0.2f}'.format(index[i], average_precision[index[i]]))

        fig = plt.gcf()
        fig.subplots_adjust(bottom=0.25)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curves for each label \n estimator {}'.format(name))
        plt.legend(lines, labels, loc='upper right', fontsize='x-small')
        plt.savefig(save_path + '{}_Precision-recall curves.png'.format(name))
        # plt.show()
