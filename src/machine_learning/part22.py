import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import sys
import time

load_path = '../../data/DeliciousMIL/'


def find(key, dictionary):
    for k, v in dictionary.iteritems():
        if k == key:
            yield v
        elif isinstance(v, dict):
            for result in find(key, v):
                yield result
        elif isinstance(v, list):
            for d in v:
                for result in find(key, d):
                    yield result


def parse_data_for_part1(file_name):
    f = open(load_path + file_name, 'r')

    count = 0
    temp_dict = {}
    for line in f:
        parsed_line = re.sub(r'<\d+>', '', line)
        parsed_line = parsed_line.strip()
        temp_arr = parsed_line.split(' ')
        temp_arr = filter(None, temp_arr)
        temp_dict[count] = temp_arr
        count += 1
    f.close()

    df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in temp_dict.iteritems()]))
    df = df.transpose()

    return df


def parse_labels(file_name):
    pl_df = pd.read_csv(load_path + file_name, header=0, sep=' ')
    pl_df = pd.DataFrame(pl_df)

    return pl_df


def load_data(file_name):
    dataset = pd.read_csv(load_path + file_name, na_values=0)
    dataset = dataset.iloc[:, 1:]
    dataset = np.array(dataset)
    return np.nan_to_num(dataset)


def parse_data_for_part2(instances, labels, size):
    f = open(load_path + instances, 'r')
    l = open(load_path + labels, 'r')

    labels = [line for line in l]
    doc_count = 0
    insta_count = 0
    bag_of_instances = {}

    for doc in f:

        label = labels[doc_count].split(" ")
        label = label[2]  # assign reference label to target array

        parsed_line = re.split(r'<\d+>', doc)

        for item in parsed_line:
            item = item.strip()
            if item:
                words = item.split(" ")
                words = [int(word) for word in words]
                bag_of_instances[insta_count] = (doc_count, words, label)
                insta_count += 1

        if doc_count == size:
            break

        doc_count += 1

    f.close()
    l.close()

    df = pd.DataFrame.from_dict(bag_of_instances)
    df = df.transpose()

    return df


def kmeans_approach(df, k=20, size=1000):

    df = df.rename(index=str, columns={0: "Bag", 1: "Instances", 2: "label"})
    df = pd.DataFrame(df)

    reform_dict = {}
    for index, item in df.iterrows():

        temp_dict = {}
        for word in item["Instances"]:
            if int(word) in temp_dict:
                temp_dict[int(word)] += 1
            else:
                temp_dict[int(word)] = 1

        temp_dict['label'] = item['label']
        temp_dict['bag'] = item["Bag"]
        reform_dict[int(index)] = temp_dict
        # if int(index) == size:
        #     break

    print "Creating inner dataframe"
    test_df = pd.DataFrame(reform_dict)
    test_df = test_df.fillna(0)
    test_df = test_df.transpose()

    test_arr = test_df.iloc[:, :]
    test_arr = np.array(test_arr)
    X = test_arr[:, :-2]
    y = test_arr[:, -1]

    print "Fitting Kmeans"
    labels = KMeans(n_clusters=k, n_jobs=7).fit_predict(X=X)
    print "Kmeans fitted."
    print "Regather as bags."

    final_dict = {}
    final_idx = 0

    bags = test_df["bag"]
    annotation = test_df["label"]

    for label in labels:

        check = list(find('bag' + str(bags[final_idx]), final_dict))

        if check:
            # if 'bag' + str(bags[final_idx]) in check:
            label_dict[label] += 1
            # final_dict['bag'+ str(bags[final_idx])] = label_dict
            label_dict["class"] = annotation[final_idx]
            final_dict['bag' + str(bags[final_idx])].update(label_dict)

            final_idx += 1
        else:
            label_dict = {}
            for i in range(k):
                label_dict[i] = 0
            label_dict['class'] = 0
            label_dict[label] += 1
            label_dict["class"] = annotation[final_idx]
            final_dict['bag' + str(bags[final_idx])] = label_dict
            final_idx += 1

    print "Regathering finished"
    final = pd.DataFrame(final_dict)
    final = final.fillna(0)
    final = final.transpose()

    final = final.iloc[:, :]
    final = np.array(final)
    X = final[:, :-1]
    y = final[:, -1]

    return X, y


if __name__ == "__main__":

    test_df = parse_data_for_part2('test-data.dat', 'test-label.dat', 1000)
    print "test set parsed"
    train_df = parse_data_for_part2('train-data.dat', 'train-label.dat', 3000)
    print "train set parsed"

    X_test, y_test = kmeans_approach(test_df, k=20, size=100)
    print "test set transformed"

    X_train, y_train = kmeans_approach(train_df, k=20, size=100)
    print "train set transformed"

    results = {}
    estimators = [("NB", GaussianNB()), ('SVM', SVC(kernel='linear', C=1, gamma=0.1)), ("DT", DecisionTreeClassifier())]

    for name, estimator in estimators:
        estimator = estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        print classification_report(y_true=y_test, y_pred=y_pred)
        print confusion_matrix(y_true=y_test, y_pred=y_pred)
        accur = accuracy_score(y_true=y_test, y_pred=y_pred)
        print "Accuracy: ", accur
        results[name] = accur

    colors = ["red", "green", "blue"]
    plt.title("Algorithm performance in a bag of instances problem")
    plt.ylabel("Accuracy %")
    plt.bar(range(len(results)), list(results.values()), align='center', color=colors)
    plt.xticks(range(len(results)), list(results.keys()))
    plt.show()

    # plt.figure()
