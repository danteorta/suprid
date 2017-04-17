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

#  -------------------- Read vtk file --------------------
vtk_file = 'D:\\Users\\dorta\Dropbox\Stanford\Research\workspace\\test_run\DiscretizationToolkit\\output_mesh.vtk'
node_coords = pn.read_csv(vtk_file, delim_whitespace=True, skiprows=5, nrows=16650, index_col=False,
                          header=None, names=['x', 'y', 'z'])
cell_nodes = pn.read_csv(vtk_file, delim_whitespace=True, skiprows=16656, nrows=3727, index_col=False,
                         header=None, names=range(1,21))
# Transform cell_nodes into a list
nodes_array = cell_nodes.values.tolist()
nodes_list = []
[nodes_list.extend(x) for x in nodes_array]
len(nodes_list)

aux = 0
data_str = {}
for cnt in range(0 , 8282):
    n_els = nodes_list[aux]
    stt_el = aux + 1
    end_el = int(stt_el + n_els)
    print([aux, n_els, stt_el, end_el])
    data_str[cnt] = nodes_list[stt_el:end_el]
    aux = int(end_el)

nodes = pn.DataFrame.from_dict(data_str,orient='index')


16656
20384


from vtk import *
from vtk.util.numpy_support import vtk_to_numpy

# load a vtk file as input
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(vtk_file)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
data = reader.GetOutput()

flowcell_id = vtk_to_numpy(data.GetCellData().GetArray('FLOWCELL_ID'))
vtk_to_numpy(flowcell_id)
# Read scalar names
for m in range(reader.GetNumberOfScalarsInFile()):
    print(reader.GetScalarsNameInFile(m))


