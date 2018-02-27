import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import scale
from scipy.stats import chi2
from pyclust import KMedoids
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


# Import data
data = pd.read_excel('C:/Users/dorta/Dropbox/Stanford/GS 240/Homeworks/Hmk4//ilr_data.xls')
ilr_cols = ['ilr'+str(x) for x in range(1,30)]
data_ilr = data.loc[:,ilr_cols]

# -------------------------- Outlier Detection --------------------------
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

inliers = data.loc[robust_dist<thresh,:]
outliers = data.loc[robust_dist>thresh,:]

# -------------------------- Factor Analysis --------------------------
inliers_full = pd.DataFrame.from_csv('inliers_factors.csv')
data_fact = inliers_full[['factor'+str(x) for x in range(1,6)]]
X = data_fact[['factor'+str(x) for x in range(1,6)]].values

clusterer = KMedoids(4)
inliers_full['label'] = cluster_labels = clusterer.fit_predict(X)


# Plot clusters in the X Y plot
groups = inliers_full.groupby('label')
# Plot
fig, ax = plt.subplots()
ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=4, label=name)
ax.legend()
plt.show(False)












range_n_clusters = [2,3,4,5,6,7,8]
for n_clusters in range_n_clusters:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(15, 7)
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    #cluster_labels = clusterer.fit_predict(X)
    clusterer = KMedoids(n_clusters)
    cluster_labels = clusterer.fit_predict(X)


    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters =", n_clusters,
          "Avg silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette plot")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = clusterer.centers_
    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("Clustered data.")
    ax2.set_xlabel("ILR 1")
    ax2.set_ylabel("ILR 2")

    plt.suptitle(("n_clusters = %d" % n_clusters),
                 fontsize=14, fontweight='bold')

    plt.show(False)














# ---------
n_vars = len(ilr_cols)
scaled_ilr = scale(data_ilr)
fa_instance = FactorAnalysis(n_components=20).fit(scaled_ilr)
fa_loads = pd.DataFrame(fa_instance.components_,columns=ilr_cols)
explained_var = np.sum((fa_loads**2).sum(1)/n_vars)