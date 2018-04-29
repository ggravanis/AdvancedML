import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

save_path = '../../results/part3/'

dataset = pd.read_csv(save_path + 'scores.csv')

df = pd.DataFrame(dataset)

names = df['Method']
names = np.array(names)
print names

df.plot.bar(width=0.7, figsize=(10, 10))
plt.ylabel('accuracy %')
plt.title("Algorithm & balancing method benchmarcking diagram")
plt.ylim((0.4, 1))
plt.xticks(np.arange(len(names)), names, rotation=60)
plt.show()
