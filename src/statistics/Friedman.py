import pandas as pd
import math


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

        self.df = dataframe.set_index('Fold')
        self.n, self.k = self.df.shape

    def get_ranked_dataframe(self):

        return self.df.rank(axis=1)

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
