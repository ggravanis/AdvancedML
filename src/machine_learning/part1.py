import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report

load_path = '../../data/DeliciousMIL/'


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


def parse_data_for_part2(instances, labels):
    f = open(load_path + instances, 'r')
    l = open(load_path + labels, 'r')

    labels = [line for line in l]
    doc_count = 0
    insta_count = 0
    bag_of_instances = {}

    for doc in f:

        label = labels[doc_count].split(" ")
        label = label[2]

        parsed_line = re.split(r'<\d+>', doc)

        for item in parsed_line:
            item = item.strip()
            if item:
                words = item.split(" ")
                words = [int(word) for word in words]
                bag_of_instances[insta_count] = (words, label)
                insta_count += 1

        doc_count += 1

    f.close()
    l.close()

    df = pd.DataFrame.from_dict(bag_of_instances)
    df = df.transpose()

    return df


def kmeans_approach(df, k=20, size=1000):
    df = df.rename(index=str, columns={0: "Instances", 1: "label"})
    df = pd.DataFrame(df)

    reform_dict = {}
    for index, item in df.iterrows():
        print index
        temp_dict = {}
        for word in item["Instances"]:
            if int(word) in temp_dict:
                temp_dict[int(word)] += 1
            else:
                temp_dict[int(word)] = 1

        temp_dict['label'] = item['label']

        reform_dict[int(index)] = temp_dict
        if int(index) == size:
            break

    test_df = pd.DataFrame(reform_dict)
    test_df = test_df.fillna(0)
    test_df = test_df.transpose()

    test_df = test_df.iloc[:, :]
    test_df = np.array(test_df)
    X = test_df[:, :-1]
    y = test_df[:, -1]

    labels = KMeans(n_clusters=k).fit_predict(X=X)

    final_dict = {}
    final_idx = 0
    for label in labels:
        temp_dict = {label: 1, 'target': y[final_idx]}
        final_dict[final_idx] = temp_dict
        final_idx += 1

    final = pd.DataFrame(final_dict)
    final = final.fillna(0)
    final = final.transpose()

    final = final.iloc[:, :]
    final = np.array(final)
    X = final[:, :-1]
    y = final[:, -1]

    return X, y


test_df = parse_data_for_part2('test-data.dat', 'test-label.dat')
print "test set parsed"
train_df = parse_data_for_part2('train-data.dat', 'train-label.dat')
print "train set parsed"

X_train, y_train = kmeans_approach(train_df, k=20, size=1500)
print "train set transformed"

X_test, y_test = kmeans_approach(test_df, k=20, size=500)
print "test set transformed"

clf = SVC(kernel='linear', C=100)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print classification_report(y_true=y_test, y_pred=y_pred)
