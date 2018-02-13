import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from scipy.stats import chi2

# Import data
data = pd.read_excel('C:/Users/dorta/Dropbox/Stanford/GS 240/Homeworks/Hmk4//ilr_data.xls')
ilr_cols = ['ilr'+str(x) for x in range(1,30)]
data_ilr = data.loc[:,ilr_cols]

# Fit the covariances
robust_cov = MinCovDet().fit(data_ilr)
emp_cov = EmpiricalCovariance().fit(data_ilr)

# Get the Mahalanobis distances
robust_dist = np.sqrt(robust_cov.mahalanobis(data_ilr))
classic_dist = np.sqrt(emp_cov.mahalanobis(data_ilr))

# Chi squared test at p=0.025
thresh = np.sqrt(chi2.isf(0.025, len(ilr_cols)))


# Plot of the outliers
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(classic_dist[robust_dist<thresh], robust_dist[robust_dist<thresh], s=7, c='c', marker="+", label='inliers')
ax1.scatter(classic_dist[robust_dist>thresh], robust_dist[robust_dist>thresh], s=7, c='r', marker="+", label='outliers')
x = np.linspace(*ax1.get_xlim())
ax1.plot(x, x, linewidth=1, linestyle='--', color='b')
ax1.plot([0, 20], [thresh, thresh], linewidth=0.5, linestyle='--', color='r')
ax1.plot([thresh, thresh], [0, 40], linewidth=0.5, linestyle='--', color='r')
plt.legend(loc='upper left')
plt.xlabel('Manhalanobis Distance')
plt.ylabel('Robust Distance')
plt.show(False)
