import pandas as pn
from matplotlib import pyplot as pl
from sklearn.cluster import KMeans
from sklearn import tree
import decimal
import pdb
from vtk import *
from vtk.util.numpy_support import vtk_to_numpy

def rounding_wb(xx):
    """This is the rule for translating from rates to the other STDwell file time """
    if xx<1:
        return str(format(xx, '.4f'))
    else:
        return str(format(round(xx,3), '.4f'))

def read_output_batches(file_path, skip_rows, batch_sz, max_batches=1000, cols=None, verbose=False):
    """
    Import as a DataFrame the data in the ASCII files from AD-GPRS output
    In the output DataFrame, the 'time' column is the timestep
    :param file_path: Path of the file to be read
    :param skip_rows: Number of rows to skip (First batch starts after these)
    :param batch_len: Number of rows that are read per batch
    :param max_batches: (Optional) Max number of batches
    :param cols: List of columns that will be read from the file(positional)
    :param verbose: : Controls verbosity
    :return: Pandas DataFrame
    """
    # Read the initial batch
    full_data = pn.read_csv(file_path, delim_whitespace=True, skiprows=skip_rows, nrows=batch_sz, index_col=False,
                            usecols=cols)
    # pdb.set_trace()
    # Iteratively import each time step from the file
    full_data['TS'] = 0
    for t_step in range(1, max_batches):
        skip_rows = skip_rows + batch_sz + 3
        try:
            my_batch = pn.read_csv(file_path, delim_whitespace=True, skiprows=skip_rows, nrows=batch_sz,
                                   usecols=cols, index_col=False)
        except pn.io.common.EmptyDataError:
            break
        my_batch['TS'] = t_step
        full_data = full_data.append(my_batch)
        if verbose:
            print("batch {0}".format(t_step))

    return full_data


# ------------------------ KEY NUMBERS FROM SIMULATION RESULTS ------------------------
# These numbers will be modified if the mesh / completions are different
# Number of cells in the simulation. 'vars' file has batches of this size
n_cells = 6706
# Number of completions in wells. 'wells.txt' has these many cells in COMPDAT
n_completions = 265
open_completions = 5

# ----------------------------------- DEFINE PATHS -----------------------------------
it_name = 'VARPRESSMODEL_T'
model_path = 'C:/Users/dorta/Dropbox/Stanford/Research/workspace/simpler_multiple/gprs/'
modif = ''
rates_path = model_path + modif +  '{0}.rates.txt'.format(it_name)
vars_path = model_path + modif +'{0}.vars.txt'.\
    format(it_name)
fips_path = model_path + modif +'{0}.FIPS.txt'.format(it_name)
wells_path = model_path + modif +'{0}.WELLS.StdWell_W.txt'.\
    format(it_name)
trans_path = model_path + 'model\\trans.txt'
compdat_path = model_path + 'model/wells.txt'
vtk_file = 'C:/Users/dorta/Dropbox/Stanford/Research/workspace/simpler_multiple/disc/output_mesh.vtk'

# ------------------------------------------------------------------------------------
# ----- Read well cells IDs from COMPDAT keyword -----
# Careful!! there is a shift in id_perf. Dunno why the file in "wells_path"
# has a different id than the COMPDAT wells.txt file
compdat = pn.read_csv(compdat_path, delim_whitespace=True, skiprows=4, nrows=n_completions,
                      index_col=False,header=None, usecols=[1,5])
compdat.columns = ['cdat','status']
compdat['id_perf'] = (compdat.cdat - 1).astype(int)

# ----- Extract variables for all cells (Full simulation results. 'vars.txt' file) -----
stt_df = read_output_batches(vars_path, skip_rows=3, batch_sz=n_cells, cols=[0,1,2], verbose=True)
wellbore_pt = stt_df.loc[compdat.id_perf]

# Add real time values (If tstep is >= 1e-4 this will work)
tsteps = pn.read_csv(rates_path, delim_whitespace=True, index_col=False, usecols=['TS','Day'],
                     dtype={'TS':int, 'Day':str})
wellbore_pt = wellbore_pt.merge(tsteps, how='left', on='TS')

# Read completion rates 'WELLS.StdWell_W' file
rates_df = read_output_batches(wells_path, skip_rows=1, batch_sz=open_completions,verbose=True)

# Join P,T data (full) with rates data (incomplete) This could be kept separated
wellbore_data = wellbore_pt.merge(rates_df, how='left', left_on=['IB', 'TS'], right_on=['PERF_NB','TS'])

# At this point wellbore_data has P, T for all cells and Rates for all the open cells
# This means the rate columns have a lot of NaNs

# ----------------- Read Spatial coordinates (vtk file) -----------------

# load a vtk file as input
reader = vtk.vtkUnstructuredGridReader()
reader.SetFileName(vtk_file)
reader.ReadAllVectorsOn()
reader.ReadAllScalarsOn()
reader.Update()
data = reader.GetOutput()

# Map of cell_id to flowcell_id
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
fcells.loc[fcells.flowcell_id.isin(wb_ids), 'in_wellbore'] = 1

new_wellbore_data = wellbore_data.merge(fcells, how='left', left_on='IB', right_on='flowcell_id')
new_wellbore_data['x'] = new_wellbore_data.x.astype(int)

# new_wellbore_data['Day'] = new_wellbore_data.Day.apply(rounding_wb,1)
# new_wellbore_data = new_wellbore_data.merge(rates_df, how='left', on=['IB','Day'])
# new_wellbore_data.fillna(0, inplace=True)


