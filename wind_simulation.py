from wind_resource import *
import pandas as pn
import timeit



# Some code
# cd D:\Users\dorta\Dropbox\Stanford\Research\workspace\suprid

wb_params = {'1': {'scale': 8.4956, 'shape': 2.1869, 'mean': 7.2766, 'max': 28.2976},
             '2': {'scale': 6.8118, 'shape': 2.0155, 'mean': 5.6486, 'max': 24.1849},
             '3': {'scale': 7.1176, 'shape': 2.2645, 'mean': 6.1673, 'max': 28.2976},
             '4': {'scale': 7.4848, 'shape': 2.1187, 'mean': 6.2753, 'max': 46.3133},
             '5': {'scale': 7.0343, 'shape': 2.0915, 'mean': 5.9080, 'max': 23.6484}
             }

wb_params = {'5': {'scale': 7.0343, 'shape': 2.0915, 'mean': 5.9080, 'max': 23.6484}}



# Define the length of the simulation
sim_days = 0.1
sim_minutes = sim_days * 24 * 60
save_folder = "D:/Users/dorta/Dropbox/Stanford/Energy 293C/data/"

for hist_n in wb_params.keys():
    start_time = timeit.default_timer()
    wb_ps = wb_params[hist_n]
    my_wind = WindResource(mean_speed=wb_ps['mean'], max_speed=wb_ps['max'], shape=wb_ps['shape'],
                           scale=wb_ps['scale'], time_delta=1)
    tseries = [my_wind.getNext() for a in range(sim_minutes)]
    ts = pn.DataFrame(tseries)
    # ts.to_csv(save_folder + '{0}.csv'.format(hist_n))
    # code you want to evaluate
    elapsed = timeit.default_timer() - start_time
    print(elapsed)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

#
# import scipy.io
# mat = scipy.io.loadmat("D:/Users/dorta/Dropbox/Stanford/Energy 293C/data/EMK_wind")
# # min((direction - direction.shift(-1)).describe()
# direction = pn.DataFrame(mat['direction_EMK'])
# timestamp = pn.DataFrame(mat['date_EMK'])
import pandas as pn
from datetime import datetime
import numpy as np

def kappa(tol_angle, degrees=True):
    """
    Receive a tolerance angle to be assumed 2 standard deviations in a normal distribution.
    Return the equivalent kappa. This is approximated when kappa is large
    :param tol_angle: Angle in degrees
    :param degrees: If True the function assume tol_angle is in degrees
    :return: kappa
    """
    if degrees:
        tol_angle = np.radians(tol_angle)
    return 4/(tol_angle**2)


pth = "D:/Users/dorta/Dropbox/Stanford/Energy 293C/data/HHV.txt"

# Read the data
data = pn.read_csv(pth, skiprows=5, sep='\t', nrows=5000)
# Some data cleaning
data.columns = [x.strip() for x in data.columns]
data.loc[:, 'drct'].replace('M', np.nan, inplace=True)
data.loc[:, 'drct'] = data.drct.astype('float')
data.dropna(axis=0, subset=['drct'], inplace=True)
data.loc[:, 'dtime'] = data.valid.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M'))
data.drop_duplicates(subset='dtime', inplace=True)
data.set_index('dtime', inplace=True, drop=False)

# Convert to radians
data.loc[:, 'orig_dir'] = np.radians(data.drct)
min_data = pn.DataFrame(data.loc[:, 'orig_dir'].resample('1T').ffill())
min_data['orig_dir_deg'] = np.degrees(min_data.orig_dir)

# Try a bunch of kappas
for ang in range(5, 95, 5):
    new_angles = np.degrees(np.random.vonmises(min_data.orig_dir, kappa(ang)))
    # Correct when angle is negative
    new_angles[new_angles < 0] = 360 + new_angles[new_angles < 0]
    min_data.loc[:, 'dir_{}'.format(ang)] = new_angles

min_data.to_csv("D:/Users/dorta/Dropbox/Stanford/Energy 293C/data/minute_data_HHV.csv")

a = min_data.apply(lambda x: min(abs(x.orig_dir_deg - x.dir_60), 360 - abs(x.orig_dir_deg - x.dir_5) ),1)

