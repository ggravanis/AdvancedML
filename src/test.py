import re
import pandas as pd
import math
import numpy as np

load_path = '../data/DeliciousMIL/'

f = open(load_path + 'test-data.dat', 'r')

count = 0
tf_dict = {}
temp_dict = {}
for line in f:
    parsed_line = re.sub(r'<\d+>', '', line)
    parsed_line = parsed_line.strip()
    temp_arr = parsed_line.split(' ')
    temp_arr = filter(None, temp_arr)
    tf = {}
    for item in temp_arr:
        if int(item) in tf:
            tf[int(item)] += 1 / float(len(temp_arr))
        else:
            tf[int(item)] = 1 / float(len(temp_arr))
    tf_dict[count] = tf
    temp_dict[count] = temp_arr
    count += 1

f.close()

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in temp_dict.iteritems()]))
df = df.transpose()
df = pd.DataFrame(df)
print

df_m, df_n = df.shape

print df_m, df_n
idf = {}
count_dict = {}
collection = {}

for index, document in df.iterrows():
    # print document
    for word in range(8520):
        print index, word
        if word in count_dict:
            if str(word) in set(document):
                count_dict[word] += 1
                idf[word] = float(df_m) / count_dict[word]
        else:
            count_dict[word] = 0
    collection[index] = idf
    if index == 1:
        break

collection_df = pd.DataFrame.from_dict(collection)
collection_df = collection_df.transpose()
print collection_df
print

a = {1: [1, 2, 3], 2: [4, 5, 6]}
b = {1: [0, 1, 0]}

a_df = pd.DataFrame.from_dict(a)
b_df = pd.DataFrame.from_dict(b)

c_df = a_df.mul(b_df, axis=0)

print a_df
print b_df
print c_df
