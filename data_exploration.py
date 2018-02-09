import pandas as pn
from matplotlib import pyplot as pl
from sklearn.cluster import KMeans
from sklearn import tree
import decimal

# ---- Define a few paths ----
it_name = 'DANTEMODEL_T'
# model_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/test_run/gprs_repo/'
model_path = 'C:/Users/dorta/Dropbox/Stanford/Research/workspace/simpler_multiple/gprs/'
modif = 'rate_50/'
rates_path = model_path + modif +  '{0}.rates.txt'.format(it_name)
vars_path = model_path + modif +'{0}.vars.txt'.\
    format(it_name)
fips_path = model_path + modif +'{0}.FIPS.txt'.format(it_name)
wells_path = model_path + modif +'{0}.WELLS.StdWell_W.txt'.\
    format(it_name)
trans_path = model_path + 'model\\trans.txt'
compdat_path = model_path + 'model/wells.txt'

# --- Extract all the simulation results (All cells) ----
# CAREFUL - THIS NUMBER MAY CHANGE
batch_len = 6706
skrow = 3
stt_df = pn.read_csv(vars_path, delim_whitespace=True, skiprows=skrow, nrows=batch_len, index_col=False,
                     usecols=['IB', 'p', 'T'])
stt_df['time'] = 0
# Iteratively import each time step from the file
for t in range(1, 500):
    skrow = skrow + batch_len + 3
    try:
        new_df = pn.read_csv(vars_path, delim_whitespace=True, skiprows=skrow, nrows=batch_len,
                             index_col=False, usecols=[0,1,2])
    except pn.io.common.EmptyDataError:
        break

    new_df['time'] = t
    stt_df = stt_df.append(new_df)
    print("batch {0}".format(t))
# Save stt_df to pickle file
# stt_df.to_pickle(model_path + "stt_df")

# read well from compdat
compdat = pn.read_csv(compdat_path, delim_whitespace=True, skiprows=4, nrows=264, index_col=False,header=None)
compdat.columns = ['1','cdat','2','3','4','5','6','7','8', '9']
compdat['id_perf'] = (compdat.cdat - 1).astype(int)

# -- Read well cells from file ----
well = pn.read_csv(wells_path, delim_whitespace=True, skiprows=1, nrows=253, index_col=False)
# well = pn.read_csv(wells_path, delim_whitespace=True, skiprows=1 ,nrows=200, index_col=False)
# Careful!! there is a shift in id_perf. Dunno why the file in "wells_path"
# has a different id than the COMPDAT wells.txt file
id_perf = compdat.id_perf
# id_perf = well.PERF_NB
wellbore_data = stt_df.loc[id_perf]
wellbore_data['temperature'] = wellbore_data['T'] - 273
wellbore_data.to_pickle(model_path + "wellbore_data")
# temp_series = wellbore_data.pivot(values='T', index='time', columns='IB')
# Add real time values (If tstep is >= 1e-4 this will work)
tsteps = pn.read_csv(rates_path, delim_whitespace=True, index_col=False, usecols=['TS','Day'])
wellbore_data = wellbore_data.merge(tsteps, how='left', right_on='TS', left_on='time')

temp_series = wellbore_data.pivot(values='T', index='IB', columns='Day')
p_temp_series = wellbore_data.pivot(values='p', index='IB', columns='Day')
# Randomize the order
# ts_2  = temp_series.sample(len(temp_series))
# temp_series = stt_df.pivot(values='T', index='IB', columns='time')

# Read rates///////////////////////////////// ------------------------
batch_len = 5
skrow = 1
rates_df = pn.read_csv(wells_path, delim_whitespace=True, skiprows=skrow, nrows=batch_len, index_col=False)
# Iteratively import each time step from the file
rates_df['time'] = 0
for t in range(1, 500):
    skrow = skrow + batch_len + 3
    try:
        new_r_df = pn.read_csv(wells_path, delim_whitespace=True, skiprows=skrow, nrows=batch_len, index_col=False)
    except pn.io.common.EmptyDataError:
        break
    new_r_df['time'] = t
    rates_df = rates_df.append(new_r_df)
    print("batch {0}".format(t))
# The skipped tsteps are: TS = [43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54] (while Rate = 0 )
# mega brute force patch
# a.append(pn.DataFrame(np.zeros([12,11]), columns=a.columns),ignore_index=True)
# Get the Day for the ratesNasty brute force way
wells_df = pn.read_csv(wells_path)
time_locs = range(6,len(wells_df),7)
# List of strings with the time after zero
aux = wells_df['Time = 0'].loc[time_locs].values.tolist()
rate_times= pn.DataFrame({'Day': [0] + [x.replace('Time = ','') for x in aux],
                          'time': range(len(aux)+1)}
                         )
# Merge with actual rates
rates_df = rates_df.merge(rate_times, how='left', on='time')
rates_df.rename(columns={'PERF_NB':'IB'},inplace=True)
# rates_df['Day2'] = rates_df.Day.apply(lambda xx: '%s' % float('%.4g' % round(xx,5)))
rates_df['Day'] = rates_df.Day.apply(lambda xx:
                                      str(decimal.Decimal(xx).quantize(decimal.Decimal('0.0000'),
                                                                   rounding=decimal.ROUND_HALF_DOWN)))
def rounding_wb(xx):
    """This is the rule for translating from rates to the other STDwell file time """
    if xx<1:
        return str(format(xx, '.4f'))
    else:
        return str(format(round(xx,3), '.4f'))


# -------------------
# # # Primary plot
# ordered_cols = id_perf.values.tolist()
# b = temp_series.loc[ordered_cols]
# b_p = p_temp_series.loc[ordered_cols]
# # pl.pcolor(temp_series.T.values)
# pl.pcolor(b)
# pl.colorbar()
# pl.show(False)
#
#
# # Primary plot
# pl.pcolor(ts_2.values)
# pl.colorbar()
# pl.show(False)

# ----------------- Read vtk fancy way -----------------
from vtk import *
from vtk.util.numpy_support import vtk_to_numpy
import pandas as pn


vtk_file = 'C:/Users/dorta/Dropbox/Stanford/Research/workspace/simpler_multiple/disc/output_mesh.vtk'

# load a vtk file as input
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(vtk_file)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
data = reader.GetOutput()

# Map of cell_id to flowcell_id
# flowcell_ids = pn.DataFrame(vtk_to_numpy(data.GetCellData().GetArray('FLOWCELL_ID')), columns=['flowcell_id'])
# flowcell_ids['cell_id'] = flowcell_ids.index
fcell_centroid = pn.DataFrame(vtk_to_numpy(data.GetCellData().GetArray('GEOM_UTIL_CENTROID')), columns=['x','y','z'])
fcells = pn.DataFrame({'flowcell_id': vtk_to_numpy(data.GetCellData().GetArray('FLOWCELL_ID')),
                       'x': fcell_centroid['x'],
                       'y': fcell_centroid['y'],
                       'z': fcell_centroid['z'],
                       'poro': vtk_to_numpy(data.GetCellData().GetArray('PORO')),
                       'gridnum': vtk_to_numpy(data.GetCellData().GetArray('GRIDNUM'))
                       })
wb_ids = compdat.id_perf.unique()
fcells.loc[:,'wellbore'] = 0
fcells.loc[fcells.flowcell_id.isin(wb_ids), 'wellbore'] = 1

new_wellbore_data = wellbore_data.merge(fcells, how='left', left_on='IB', right_on='flowcell_id')
new_wellbore_data['x'] = new_wellbore_data.x.astype(int)

new_wellbore_data['Day'] = new_wellbore_data.Day.apply(rounding_wb,1)
new_wellbore_data = new_wellbore_data.merge(rates_df, how='left', on=['IB','Day'])
new_wellbore_data.fillna(0, inplace=True)


# ---------end
new_wellbore_data.loc[new_wellbore_data.IB.isin([4374, 2737, 2472, 1031, 1225]),:]

iphone = new_wellbore_data.groupby(['Day', 'x']).T.mean().reset_index()
new_temp_series = iphone.pivot(values='T', index='x', columns='Day')
# wellbore_centroids.loc[wellbore_centroids.x.between(-1,1) & wellbore_centroids.z.between(-701, -698)]
# new_temp_series[70].plot()


pl.pcolor(new_temp_series)
pl.colorbar()
pl.show(False)

new_temp_series[10].plot()
new_temp_series[30].plot()
new_temp_series[50].plot()
new_temp_series[70].plot()
new_temp_series[85].plot()
pl.show(False)


# Plot centroids
ax = fcells.loc[fcells.wellbore==1].plot.scatter(x='x', y='z', color='red', label='wellbore');
fcells.loc[fcells.wellbore==0].plot.scatter(x='x', y='z', color='gray', label='other', ax=ax);
fcells.loc[fcells.flowcell_id==4626].plot.scatter(x='x', y='z', color='blue', label='other', ax=ax);
fcells.loc[fcells.flowcell_id==4612].plot.scatter(x='x', y='z', color='blue', label='other', ax=ax);
fcells.loc[fcells.flowcell_id==2473].plot.scatter(x='x', y='z', color='blue', label='other', ax=ax);
fcells.loc[fcells.flowcell_id==2739].plot.scatter(x='x', y='z', color='blue', label='other', ax=ax);
fcells.loc[fcells.flowcell_id==1034].plot.scatter(x='x', y='z', color='blue', label='other', ax=ax);
fcells.loc[fcells.flowcell_id==1223].plot.scatter(x='x', y='z', color='blue', label='other', ax=ax);
fcells.loc[fcells.flowcell_id==4852].plot.scatter(x='x', y='z', color='green', label='other', ax=ax);
fcells.loc[fcells.flowcell_id==4374].plot.scatter(x='x', y='z', color='green', label='other', ax=ax);

pl.show(False)


# # Coordinates of each node
# pts_coords = pn.DataFrame(vtk_to_numpy(data.GetPoints().GetData()), columns=['x', 'y', 'z'])
# pts_coords['node_id'] = pts_coords.index
#
# # cell_locations = index where the integer with the number of nodes in each cell is located
# # cell_data[cell_locations] <-- Number of nodes in each cell
# cell_locations = vtk_to_numpy(data.GetCellLocationsArray())
# cell_data = vtk_to_numpy(data.GetCells().GetData())
#
# nodes_dict = []
# for c in range(len(cell_locations)):
#     stt = cell_locations[c] + 1
#     # last case
#     if c == len(cell_locations)-1:
#         pt_ids = cell_data[stt:]
#     else:
#         end = cell_locations[c+1]
#         pt_ids = cell_data[stt:end]
#     nodes_dict.append(pt_ids)

# all_nodes = pn.DataFrame(nodes_dict).stack().reset_index()
# all_nodes.rename(columns={'level_0': 'cell_id', 'level_1': 'cell_node_count', 0: 'node_id'}, inplace=True)
# all_nodes.loc[:, ['cell_id', 'node_id']] = all_nodes.loc[:, ['cell_id', 'node_id']].astype(int)
#
# all_nodes = all_nodes.merge(pts_coords, on='node_id', how='left')
#
# # Get centroids
# centroids = all_nodes.loc[:,['cell_id','x','y','z']].groupby('cell_id').agg('mean').reset_index()
# centroids = centroids.merge(flowcell_ids, on='cell_id', how='left')

# These are the correct ones
# wb_ids = wellbore_data.IB.unique()

centroids.loc[:,'wellbore'] = 0
centroids.loc[centroids.flowcell_id.isin(wb_ids), 'wellbore'] = 1
wellbore_centroids = centroids.loc[centroids.wellbore==1]

# /////Get the right coords  in temp_series
new_wellbore_data = wellbore_data.merge(wellbore_centroids, how='left', left_on='IB', right_on='flowcell_id')
new_wellbore_data['x'] = new_wellbore_data.x.astype(int)


iphone = new_wellbore_data.groupby(['time', 'x']).T.mean().reset_index()
new_temp_series = iphone.pivot(values='T', index='x', columns='time')
wellbore_centroids.loc[wellbore_centroids.x.between(-1,1) & wellbore_centroids.z.between(-701, -698)]
new_temp_series[70].plot()


pl.pcolor(new_temp_series)
pl.colorbar()
pl.show(False)

# Scatter plot of centroids
ax = centroids.loc[centroids.wellbore==1].plot.scatter(x='x', y='z', color='red', label='wellbore');
centroids.loc[centroids.wellbore==0].plot.scatter(x='x', y='z', color='gray', label='other', ax=ax);
pl.show(False)


wb = wellbore_data.merge(centroids, how='left', left_on='IB', right_on='flowcell_id')
wb.loc[:, 'T'] = wellbore_data.loc[:, 'T'] - 273
temp_series = wb.pivot(values='T', index='x', columns='time').sort_index()
pl.pcolor(temp_series.values)
pl.colorbar()
pl.show(False)


a = wb.drop_duplicates(subset=['flowcell_id'])
a.plot.scatter(x='x', y='y')

# # Read scalar names
# for m in range(reader.GetNumberOfScalarsInFile()):
#     print(reader.GetScalarsNameInFile(m))
#
#
# import matplotlib.pyplot as plt
# import numpy as np
# column_labels = list('ABCD')
# row_labels = list('WXYZ')
# data = temp_series.values
# fig, ax = plt.subplots()
# heatmap = ax.pcolor(data)
#
# # put the major ticks at the middle of each cell, notice "reverse" use of dimension
# y = np.array(temp_series.index.tolist())
# x = np.array(temp_series.columns.tolist())
# ax.set_yticks(x, minor=False)
# ax.set_xticks(y, minor=False)
#
#
# ax.set_xticklabels(x, minor=False)
# ax.set_yticklabels(y, minor=False)
# plt.show()


# ---------------------- Machine- learning part ----------------------------------------------


# Add some normalized columns
wellbore_data['time_norm'] = wellbore_data['time'].astype(float) /\
                             (wellbore_data['time'].max() - wellbore_data['time'].min())
wellbore_data['temp_norm'] = (wellbore_data['T'] - wellbore_data['T'].min()) / \
                             (wellbore_data['T'].max() - wellbore_data['T'].min())
wellbore_data['len_norm'] = (wellbore_data.IB.astype(float) - wellbore_data.IB.min()) / \
                           (wellbore_data.IB.max() - wellbore_data.IB.min())
X = wellbore_data.loc[:, ['time_norm', 'temp_norm', 'len_norm']].values

# Clustering
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
wellbore_data.loc[wellbore_data.IB.isin([320,5654,5656]), 'fracture'] = 1
X = wellbore_data.loc[:, ['time_norm', 'temp_norm']]

Y = wellbore_data.loc[:,'fracture']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
wellbore_data['fracture_pred'] = clf.predict(X)

ts_2  = temp_series.sample(len(temp_series))
X_test = ts_2.stack().reset_index()
X_test.rename(columns={0:'T', 'level_0': 'IB'},inplace=True)
X_test['time_norm'] = X_test['time'].astype(float) / (X_test['time'].max() - X_test['time'].min())
X_test['temp_norm'] = (X_test['T'] - X_test['T'].min()) / (X_test['T'].max() - X_test['T'].min())
X_test['fracture_pred'] = clf.predict(X_test.loc[:, ['time_norm', 'temp_norm']])

ts_3 = wellbore_data.pivot(values='fracture_pred', index='IB', columns='time')
ts3 = X_test.pivot(values='fracture_pred', index='IB', columns='time')
pl.pcolor(ts3.values)
pl.colorbar()
pl.show(False)


# Read transmissibilities

well = pn.read_csv(wells_path, delim_whitespace=True, skiprows=1, index_col=False)

