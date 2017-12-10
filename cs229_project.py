import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def two_pt_deriv(tseries):
    """
    Compute a columnwise two point derivative
    :param tseries: DataFrame, each column will be computed independently
    :return: Dataframe, same size as input
    """
    dy = tseries.diff(2).shift(-1)
    dx = pd.Series(tseries.index).diff(2).shift(-1)
    return dy.apply(lambda x: x.values / dx.values, axis=0)


# Import data
folder = 'C:/Users/dorta/Dropbox/Stanford/CS 229/project/'
data = pd.read_pickle(folder + 'wellbore_data_half_two_prod')

# Add noise
max_noise = 0.5
noise = (np.random.random(len(data))-0.5) * max_noise
data['T_noisy'] = data['T'] + noise

# Pivot to time-series
grouping = data.groupby(['Day', 'x']).T_noisy.mean().reset_index()
t_series = grouping.pivot(values='T_noisy', index='x', columns='Day')

# Plot some examples of tseries
# t_series[[0.028, 0.5, 0.9014, 1.5572, 2.0525, 2.5]].plot(legend=True)
# plt.show(False)

# Compute derivatives (two point formula)
dTdx = two_pt_deriv(t_series)
dTdt = two_pt_deriv(t_series.T).T

data_perf = data.loc[data.IB.isin([4374, 2737, 2472, 1031, 1225]),:]