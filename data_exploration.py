import pandas as pn
from matplotlib import pyplot as pl
from sklearn.cluster import KMeans
from sklearn import tree

# ---- Define a few paths ----
it_name = 'DANTEMODEL_T'
rates_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/test_run/gprs_repo/{0}.rates.txt'.format(it_name)
vars_path = 'D:\\Users\\dorta\\Dropbox\\Stanford\\Research\\workspace\\test_run\\gprs_repo\\{0}.vars.txt'.\
    format(it_name)
fips_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/test_run/gprs_repo/{0}.FIPS.txt'.format(it_name)
wells_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/test_run/gprs_repo/{0}.WELLS.StdWell_W.txt'.\
    format(it_name)
trans_path = 'D:\\Users\dorta\Dropbox\Stanford\Research\workspace\\test_run\gprs_repo\model\\trans.txt'

# --- Extract all the simulation results (All cells) ----
# CAREFUL - THIS NUMBER MAY CHANGE
batch_len = 8282
skrow = 3
stt_df = pn.read_csv(vars_path, delim_whitespace=True, skiprows=skrow, nrows=8282, index_col=False)
stt_df['time'] = 0
# Iteratively import each time step from the file
for t in range(1, 59):
    new_df = pn.read_csv(vars_path, delim_whitespace=True, skiprows=skrow, nrows=8282, index_col=False)
    new_df['time'] = t
    stt_df = stt_df.append(new_df)
    skrow = skrow + batch_len + 3
    print("batch {0}".format(t))


# -- Read well cells from file ----
well = pn.read_csv(wells_path, delim_whitespace=True, skiprows=1, nrows=53, index_col=False)
# Careful!! there is a shift in id_perf. Dunno why the file in "wells_path"
# has a different id than the COMPDAT wells.txt file
id_perf = well.PERF_NB
wellbore_data = stt_df.loc[id_perf]
wellbore_data['temperature'] = wellbore_data['T'] - 373
temp_series = wellbore_data.pivot(values='T', index='IB', columns='time')

# temp_series = stt_df.pivot(values='T', index='IB', columns='time')


# Primary plot
pl.pcolor(temp_series.values)
pl.colorbar()
pl.show(False)


# Add some normalized columns
wellbore_data['time_norm'] = wellbore_data['time'].astype(float) /\
                             (wellbore_data['time'].max() - wellbore_data['time'].min())
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
X = wellbore_data.loc[:, ['time_norm', 'temp_norm']]
Y = wellbore_data.loc[:,'fracture']
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
wellbore_data['fracture_pred'] = clf.predict(X)

ts3 = wellbore_data.pivot(values='fracture', index='IB', columns='time')
pl.pcolor(ts3.values)
pl.colorbar()
pl.show(False)


# Read transmissibilities

well = pn.read_csv(wells_path, delim_whitespace=True, skiprows=1, index_col=False)




# ----------------- Read vtk fancy way -----------------
from vtk import *
from vtk.util.numpy_support import vtk_to_numpy
import pandas as pn

vtk_file = 'D:\\Users\\dorta\Dropbox\Stanford\Research\workspace\\test_run\DiscretizationToolkit\\output_mesh.vtk'

# load a vtk file as input
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(vtk_file)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
data = reader.GetOutput()

# Map of cell_id to flowcell_id
flowcell_ids = pn.DataFrame(vtk_to_numpy(data.GetCellData().GetArray('FLOWCELL_ID')), columns=['flowcell_id'])
flowcell_ids['cell_id'] = flowcell_ids.index
# Coordinates of each node
pts_coords = pn.DataFrame(vtk_to_numpy(data.GetPoints().GetData()), columns=['x', 'y', 'z'])
pts_coords['node_id'] = pts_coords.index

# cell_locations = index where the integer with the number of nodes in each cell is located
# cell_data[cell_locations] <-- Number of nodes in each cell
cell_locations = vtk_to_numpy(data.GetCellLocationsArray())
cell_data = vtk_to_numpy(data.GetCells().GetData())

nodes_dict = []
for c in range(len(cell_locations)):
    stt = cell_locations[c] + 1
    # last case
    if c == len(cell_locations)-1:
        pt_ids = cell_data[stt:]
    else:
        end = cell_locations[c+1]
        pt_ids = cell_data[stt:end]
    nodes_dict.append(pt_ids)

# Dataframe with the ids of the nodes corresponding to each cell
all_nodes = pn.DataFrame(nodes_dict).stack().reset_index()
all_nodes.rename(columns={'level_0': 'cell_id', 'level_1': 'cell_node_count', 0: 'node_id'}, inplace=True)
all_nodes.loc[:, ['cell_id', 'node_id']] = all_nodes.loc[:, ['cell_id', 'node_id']].astype(int)

all_nodes = all_nodes.merge(pts_coords, on='node_id', how='left')

# Get centroids
centroids = all_nodes.loc[:,['cell_id','x','y','z']].groupby('cell_id').agg('mean').reset_index()
centroids = centroids.merge(flowcell_ids, on='cell_id', how='left')

# These are the correct ones
wb_ids = wellbore_data.IB.unique()

centroids.loc[:,'wellbore'] = 0
centroids.loc[centroids.flowcell_id.isin(wb_ids), 'wellbore'] = 1


ax = centroids.loc[centroids.wellbore==1].plot.scatter(x='x', y='y', color='red', label='wellbore');

centroids.loc[centroids.wellbore==0].plot.scatter(x='x', y='y', color='gray', label='other', ax=ax);


wb = wellbore_data.loc[:, ['T', 'IB', 'time']].merge(centroids, how='left', left_on='IB', right_on='flowcell_id')
wb.loc[:, 'T'] = wellbore_data.loc[:, 'T'] - 273
temp_series = wb.pivot(values='T', index='x', columns='time').sort_index()
pl.pcolor(temp_series.values)
pl.colorbar()
pl.show(False)


a = wb.drop_duplicates(subset=['flowcell_id'])
a.plot.scatter(x='x', y='y')

# Read scalar names
for m in range(reader.GetNumberOfScalarsInFile()):
    print(reader.GetScalarsNameInFile(m))


# #  -------------------- Read vtk file brute force --------------------
# vtk_file = 'D:\\Users\\dorta\Dropbox\Stanford\Research\workspace\\test_run\DiscretizationToolkit\\output_mesh.vtk'
# node_coords = pn.read_csv(vtk_file, delim_whitespace=True, skiprows=5, nrows=16650, index_col=False,
#                           header=None, names=['x', 'y', 'z'])
# cell_nodes = pn.read_csv(vtk_file, delim_whitespace=True, skiprows=16656, nrows=3727, index_col=False,
#                          header=None, names=range(1,21))
# # Transform cell_nodes into a list
# nodes_array = cell_nodes.values.tolist()
# nodes_list = []
# [nodes_list.extend(x) for x in nodes_array]
# len(nodes_list)
#
# aux = 0
# data_str = {}
# for cnt in range(0 , 8282):
#     n_els = nodes_list[aux]
#     stt_el = aux + 1
#     end_el = int(stt_el + n_els)
#     print([aux, n_els, stt_el, end_el])
#     data_str[cnt] = nodes_list[stt_el:end_el]
#     aux = int(end_el)
#
# nodes = pn.DataFrame.from_dict(data_str,orient='index')
#
#