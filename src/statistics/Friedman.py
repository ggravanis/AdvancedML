import pandas as pd
import math
import matplotlib.pyplot as plt

class Friedman:

    def __init__(self, dataframe, p_value):

        self.dataframe = dataframe

        if p_value:
            self.p_value = p_value
        else:
            self.p_value = 0.05

        if isinstance(self.dataframe, pd.DataFrame):
            pass
        else:
            raise TypeError("Expecting a DataFrame here. Instead I got a %s" % type(self.dataframe))

        valid = {0.1, 0.05, 0.01}
        if p_value not in valid:
            raise ValueError("results: status must be one of %r." % valid)

        self.df = dataframe.set_index('Folds')
        self.n, self.k = self.df.shape

    def get_ranked_dataframe(self):

        return self.df.rank(axis=1, ascending=False)

    def get_rank_mean_values(self):
        df_rank = self.get_ranked_dataframe()
        df_rank = df_rank.mean(axis=0)
        mean_values = df_rank.values
        return mean_values

    def get_sum_of_values_squared(self):
        values_squared = [value * value for value in self.get_rank_mean_values()]
        return sum(values_squared)

    def get_chi_squared(self):
        chi_squared = 12.0 * self.n * (
                self.get_sum_of_values_squared() - (self.k * (self.k + 1) * (self.k + 1)) / 4) / (
                              self.k * (self.k + 1))
        return chi_squared

    def get_F_value(self):
        F_value = (self.n - 1) * self.get_chi_squared() / (self.n * (self.k - 1) - self.get_chi_squared())
        return F_value

    def get_Critical_Difference(self):

        q_a_001 = [2.576, 2.913, 3.113, 3.255, 3.364, 3.452, 3.526, 3.590, 3.646, 3.696, 3.741, 3.781, 3.818, 3.853,
                   3.884, 3.914, 3.941, 3.967, 3.992]
        q_a_005 = [1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031, 3.102, 3.164, 3.219, 3.268, 3.313, 3.354, 3.391,
                   3.426, 3.458, 3.489, 3.517, 3.544]
        q_a_010 = [1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780, 2.855, 2.920, 2.978, 3.030, 3.077, 3.120, 3.159,
                   3.196, 3.230, 3.261, 3.291, 3.319]

        if self.k > (len(q_a_001)+1):
            raise ValueError("I can calculate Critical Difference for up to 20 algorithms")

        if self.p_value == 0.05:
            q_a = q_a_005[self.k - 2]
        elif self.p_value == 0.1:
            q_a = q_a_010[self.k - 2]
        elif self.p_value == 0.01:
            q_a = q_a_001[self.k - 2]
        else:
            q_a = 0

        critical_difference = q_a * math.sqrt(self.k * (self.k + 1) / (6.0 * self.n))
        return critical_difference

    def plot_test(self):

        ranked = self.get_ranked_dataframe()

        m_r, n_r = ranked.shape
        crt_difference = self.get_Critical_Difference()
        print crt_difference
        y_height = 1.2
        x_offset = 3
        temp = {'X': self.get_rank_mean_values(), 'Names': ranked.keys(), 'y': [y_height for i in range(n_r)]}

        df = pd.DataFrame(temp)

        df = df.sort_values(by='X')
        df = df.reset_index(drop=True)

        m, n = df.shape

        y_pos = []
        x_pos = []

        y_left_pos = y_height - 0.5
        y_right_pos = y_height

        for index, row in df.iterrows():
            print
            if row['X'] <= m / 2.0:
                y_left_pos -= 0.1
                y_pos.append(y_left_pos)
                x_pos.append(row['X'] - x_offset)
            if row['X'] > m / 2.0:
                y_right_pos -= 0.1
                y_pos.append(y_right_pos)
                x_pos.append(row['X'] + x_offset)

        df['y_pos'] = y_pos
        df['x_pos'] = x_pos

        fig = plt.figure(figsize=(15, 10))
        plt.scatter(df['X'], df['y'], c='black')

        plt.plot([0, m], [y_height, y_height], 'black')
        plt.ylim(0, 2)
        for index1, row in df.iterrows():
            if row['X'] <= m / 2.0:
                plt.plot([row['X'], row['X']], [row['y'], row['y_pos']], 'black')
                plt.plot([row['X'], row['x_pos']], [row['y_pos'], row['y_pos']], 'black')

                plt.annotate(row['Names'] + " (" + str(round(row['X'], 2)) + ")", (row['x_pos'], row['y_pos']),
                             xytext=(row['x_pos'], row['y_pos'] + 0.01))

            if row['X'] > m / 2.0:
                plt.plot([row['X'], row['X']], [row['y'], y_height - row['y_pos']], 'black')
                plt.plot([row['X'], row['x_pos']], [y_height - row['y_pos'], y_height - row['y_pos']], 'black')
                plt.annotate(row['Names'] + " (" + str(round(row['X'], 2)) + ")",
                             (row['x_pos'], y_height - row['y_pos']),
                             xytext=(row['x_pos'] + 0.5, y_height - row['y_pos'] + 0.01))

        # Add Critical Difference plot

        for major_tick in range(m + 1):
            minor_tick = major_tick + 0.5

            if minor_tick <= m:
                print minor_tick, m
                plt.plot([minor_tick, minor_tick], [y_height, y_height + 0.05], color='black')

            plt.plot([major_tick, major_tick], [y_height, y_height + 0.1], color='black')
            plt.annotate(major_tick, (major_tick, y_height + 0.1), xytext=(major_tick + 0.1, y_height + 0.12))

        plt.plot([m, m - crt_difference], [1.5, 1.5], 'black')
        plt.plot([m, m], [1.48, 1.52], 'black')
        plt.plot([m - crt_difference, m - crt_difference], [1.48, 1.52], 'black')
        plt.annotate("CD = {}".format(round(crt_difference, 2)), (0, 1.5), xytext=(m - 0.5, 1.55))

        plt.gca().invert_xaxis()
        plt.axis('off')

        return plt