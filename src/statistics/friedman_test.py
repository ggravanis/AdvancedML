import pandas as pd
from scipy.stats import friedmanchisquare


# Load the data
load_path = '../../results/part1/'
# save_path = '../../results/part1/'


data = pd.read_csv(load_path + 'accuracy_benchmarking.csv')
df = pd.DataFrame(data=data)


fried_value, fried_p = friedmanchisquare(df['AdaBoost'], df['Bagging'], df['GradientBoost'], df['RandomForest'])

print fried_value, fried_p
#

# data = data.iloc[:, 1:]
#         dataset = np.array(dataset)
#         X = dataset[:, :-1]
#         y = dataset[:, -1]