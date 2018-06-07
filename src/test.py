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
for key in range(0, 8520):
    for row in df.iterrows():
        doc = [int(i) for i in row[1] if str(i).lower != "nan"]
        print
        if key in doc and key in count_dict:
            print "bingo"
            count_dict[key] += 1
            idf[key] = math.log(float(df_m) / count_dict[key])
            continue
        else:
            count_dict[key] = 0
            print "HEdaya"

idf_df = pd.DataFrame.from_dict(idf)
print