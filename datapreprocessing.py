import pandas as pd
import numpy as np
path = '../data/'
dataset = '10'
data = pd.read_pickle(path + dataset + '.pkl')
df = pd.DataFrame(data)

"""
Datasets used:

1. Deceptive opinion spam corpus  !!!!ok!!!!
2. ironic corpus !!!ok!!!!
sentiment labelled sentences
3. yelp !!!!ok!!!
4. IMDB !!!!ok!!!
7. Amazon !!!ok!!

8. sms - spam collection dataset !!!ok!!!
9. twitter airline sentiment !!!ok!!!
YouTube Spam collection
10. 
11.
12.

"""


print df.keys()

print df.count()
print df.describe()

# mapping = {'positive': 0, 'negative': 1}
# df = df.replace(mapping)

# print df

# cols = df.columns.tolist()
df = df.rename(index=str, columns={"CONTENT": "text", "CLASS": "label"})


# cols = ['text', 'label']
# df = df[cols]
#
print df.keys()
# print df
#
# print cols
# #
print df
# print df.describe()

df.to_pickle(path + dataset + '.pkl')


