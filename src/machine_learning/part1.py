import pandas as pd
import numpy as np
import regex
import re
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

from sklearn.multiclass import OneVsOneClassifier
import matplotlib.pyplot as plt

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

    doc_dict = {}
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

        # doc_dict["doc_"+str(doc_count)] = bag_of_instances
        doc_count += 1

    f.close()
    l.close()

    df = pd.DataFrame.from_dict(bag_of_instances)
    df = df.transpose()

    return df


df = parse_data_for_part2('test-data.dat', 'test-label.dat')

df = df.rename(index=str, columns={0: "Instances", 1: "label"})

from sklearn.cluster import KMeans

y = df["label"]

X = pd.DataFrame(df['Instances'].values.tolist())




X_merged = []

for index, row in X.iterrows():
    how_many = 0
    row_sum = [item for item in row if ~np.isnan(item)]

    print float(sum(row_sum))/float(len(row_sum))
    X_merged.append(float(sum(row_sum))/float(len(row_sum)))

# X = pd.DataFrame(X).fillna(-1)

labels = KMeans(n_clusters=3).fit_predict(X=X_merged)

# estimators = [('k_means_3', KMeans(n_clusters=3)), ('k_means_8', KMeans(n_clusters=8))]
# ('k_means_8', KMeans(n_clusters=8)),
# ('k_means_3', KMeans(n_clusters=8))]

fignum = 1
# titles = ['3 clusters', '8 clusters']  # , '8 clusters', '3 clusters']

print labels

plt.figure(figsize=(15, 15))
plt.scatter(X_merged, y, c=labels)
plt.title("Incorrect Number of Blobs")
plt.show()

