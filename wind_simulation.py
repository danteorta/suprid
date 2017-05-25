from wind_resource import *
import pandas as pn
import timeit



# Some code
# cd D:\Users\dorta\Dropbox\Stanford\Research\workspace\suprid


wb_params = {'1': {'scale': 7.4081, 'shape': 1.1255, 'mean': 7.2766, 'max': 28.2976},
             '2': {'scale': 7.4081, 'shape': 1.1255, 'mean': 7.2766, 'max': 28.2976},
             '3': {'scale': 7.4081, 'shape': 1.1255, 'mean': 7.2766, 'max': 28.2976},
             '4': {'scale': 7.4081, 'shape': 1.1255, 'mean': 7.2766, 'max': 28.2976}
             }
wb_params = {'1': {'scale': 7.4081, 'shape': 1.1255, 'mean': 7.2766, 'max': 28.2976},
             }


# Define the length of the simulation
sim_days = 30
sim_minutes = sim_days * 24 * 60
save_folder = "D:/Users/dorta/Dropbox/Stanford/Energy 293C/data/"

for hist_n in wb_params.keys():
    start_time = timeit.default_timer()
    wb_ps = wb_params[hist_n]
    my_wind = WindResource(mean_speed=wb_ps['mean'], max_speed=wb_ps['max'], shape=wb_ps['shape'],
                           scale=wb_ps['scale'], time_delta=1)
    tseries = [my_wind.getNext() for a in range(sim_minutes)]
    ts = pn.DataFrame(tseries)
    ts.to_csv(save_folder + '{0}'.format(hist_n))
    # code you want to evaluate
    elapsed = timeit.default_timer() - start_time
    print(elapsed)


