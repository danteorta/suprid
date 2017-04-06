import pandas as pn
import numpy as np
from matplotlib import pyplot as pl
import sklearn

it_name = 'DANTEMODEL_T'
rates_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/test_run/gprs_repo/{0}.rates.txt'.format(it_name)
vars_path = 'D:\\Users\\dorta\\Dropbox\\Stanford\\Research\\workspace\\test_run\\gprs_repo\\{0}.vars.txt'.format(it_name)
fips_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/test_run/gprs_repo/{0}.FIPS.txt'.format(it_name)
wells_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/test_run/gprs_repo/{0}.WELLS.StdWell_W.txt'.format(it_name)

# rates = pn.read_table(rates_path, sep='\s+')
# fips = pn.read_table(fips_path, sep='\s+')
# vrs = pn.read_table(vars_path, sep='\s+', header=1)
#
# # vrs['T'] = vrs['T'].astype(float)
# breaks = vrs['T'].loc[vrs['T'].isin(['S1'])]
#
# vrs = pn.read_table(vars_path, sep='\s+', header=1)
# # vv = pn.read_hdf(vars_path, key='Time');
#
#
# vv = pn.read_csv(vars_path, delim_whitespace=True, skiprows=3, nrows=8282)

# --- Extract all the relevant features ----
batch_len = 8282
dd = {}
skrow = 3
stt_df = pn.read_csv(vars_path, delim_whitespace=True, skiprows=skrow, nrows=8282, index_col=False)
stt_df['time'] = 0
for t in range(1, 59):
    new_df = pn.read_csv(vars_path, delim_whitespace=True, skiprows=skrow, nrows=8282, index_col=False)
    # dd[t] = new_df
    new_df['time'] = t
    stt_df = stt_df.append(new_df)
    skrow = skrow + batch_len + 3
    print("batch {0}".format(t))

ts = stt_df.pivot_table(values='T', index='IB', columns='time')


# -- Read well cells from file ----
well = pn.read_csv(wells_path, delim_whitespace=True, skiprows=1, nrows=53, index_col=False)
id_perf = well.PERF_NB +1
wellbore_data = stt_df.loc[id_perf]
wellbore_data['temperature'] = wellbore_data['T'] - 373
ts = wellbore_data.pivot(values='T', index='IB', columns='time')


# ------- Waterfall Plot ---------
# methods = [None, 'none']
#
# np.random.seed(0)
# grid = ts.values
# fig, axes = pl.subplots(1,2, figsize=(12, 6),
#                          subplot_kw={'xticks': [], 'yticks': []})
#
# fig.subplots_adjust(hspace=0.3, wspace=0.05)
#
# for ax, interp_method in zip(axes.flat, methods):
#     ax.imshow(grid, interpolation='none', cmap='viridis')
#     ax.set_title(interp_method)
#
# pl.show()



from sklearn.cluster import KMeans

wellbore_data['time_norm'] = wellbore_data['time'].astype(float) / (wellbore_data['time'].max() - wellbore_data['time'].min())
wellbore_data['temp_norm'] = (wellbore_data.temperature - wellbore_data.temperature.min()) / \
                             (wellbore_data.temperature.max() - wellbore_data.temperature.min())
wellbore_data['ib_norm'] = (wellbore_data.IB.astype(float) - wellbore_data.IB.min()) / \
                           (wellbore_data.IB.max() - wellbore_data.IB.min())
X = wellbore_data.loc[:, ['time_norm', 'temp_norm', 'ib_norm']].values
kmeans = KMeans(n_clusters=3).fit(X)
wellbore_data['kmeans_label'] = kmeans.labels_

ts2 = wellbore_data.pivot(values='kmeans_label', index='IB', columns='time')
pl.pcolor(ts.values)
pl.colorbar()
pl.show(False)

# --------- clasiffier - -----
320
5654

wellbore_data['fracture'] = 0
wellbore_data.loc[wellbore_data.IB.isin([320,5654]), 'fracture'] = 1
from sklearn import tree
X = wellbore_data.loc[:, ['time_norm', 'temp_norm']]
Y = wellbore_data.loc[:,'fracture']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
wellbore_data['fracture_pred'] = clf.predict(X)

ts3 = wellbore_data.pivot(values='fracture', index='IB', columns='time')
pl.pcolor(ts3.values)
pl.colorbar()
pl.show(False)




