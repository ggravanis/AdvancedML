import pandas as pd
import numpy as np
import regex
import re
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsOneClassifier

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


y_train = load_data('train_labels.csv')
y_test = load_data('test_labels.csv')

# print y_test[0]
# le = LabelEncoder()
# y_test_enc = le.fit(y_test)
#
# print y_test_enc[0]

X_train = load_data('part1_train.csv')
X_test = load_data('part1_test.csv')
#

classif = OneVsOneClassifier(SVC(kernel='linear'))


# classif = OneVsRestClassifier(SVC(kernel='linear'))
classif.fit(X_train, y_train)

y_pred = classif.predict(X_test)

print y_pred
# print y_train[0]
# for item in y_train:
#     print item[1:]

# y_test = pd.read_csv(load_path + 'test-label.csv')
#
#
# X_train = pd.read_csv(load_path + 'part1_train.csv')
# X_test = pd.read_csv(load_path + 'part1_test.csv')
#
#
# X = dataset[:, :-1]
# y = dataset[:, -1]