import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.preprocessing import Normalizer, StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

def two_pt_deriv(tseries):
    """
    Compute a columnwise two point derivative
    :param tseries: DataFrame, each column will be computed independently
    :return: Dataframe, same size as input
    """
    dy = tseries.diff(2).shift(-1)
    dx = pd.Series(tseries.index).diff(2).shift(-1)
    return dy.apply(lambda x: x.values / dx.values, axis=0)

def tserialize(my_data, gp_by=['x','Day'], values='T_noisy'):
    grouping = my_data.groupby(gp_by)[values].mean().reset_index()
    return grouping.pivot(values=values, index=gp_by[0], columns=gp_by[1])

# ---------------------- Data Preprocessing --------------------------------
# Import data
folder = 'Stanford/CS 229/project/'
data = pd.read_pickle(folder + 'wellbore_data_3days_300_rate')
data['Day'] = data.Day.astype(float)

# Add noise
max_noise = 0.5
noise = (np.random.random(len(data))-0.5) * max_noise
data['T_noisy'] = data['T'] + noise

# Pivot to time-series
grouping = data.groupby(['Day', 'x']).T.mean().reset_index()
t_series = grouping.pivot(values='T', index='x', columns='Day')

# Compute derivatives (two point formula)
dTdx = two_pt_deriv(t_series)
dTdt = two_pt_deriv(t_series.T).T
# Return to the data df form
dTdx_unstacked = dTdx.unstack().reset_index()
dTdx_unstacked.rename(columns={0:'dTdx'}, inplace=True)
dTdt_unstacked = dTdt.unstack().reset_index()
dTdt_unstacked.rename(columns={0:'dTdt'}, inplace=True)
# Add to data
data = data.merge(dTdx_unstacked, how='left', on=['Day','x'])
data = data.merge(dTdt_unstacked, how='left', on=['Day','x'])
data_perf = data.loc[data.IB.isin([4374, 2737, 1031]),:]

data_perf = data_perf[['IB','p','T','TS','Day','poro','x','y','z','RATE_P0','dTdt', 'dTdx']]

# -------------
# Create time series of variables (Easy to visualize)
rate_series = tserialize(data, ['x','Day'],'RATE_P0')
rate = data_perf.pivot_table(values='RATE_P0',index='Day',columns='IB').reset_index()
dTdx = data_perf.pivot_table(values='dTdx',index='Day',columns='IB').reset_index()
dTdt = data_perf.pivot_table(values='dTdt',index='Day',columns='IB').reset_index()
T = data_perf.pivot_table(values='T',index='Day',columns='IB').reset_index()

srs = rate.merge(dTdx, on=['Day'],suffixes=['_rate','_dTdx'])
srs2 = dTdt.merge(T, on=['Day'], suffixes=['_dTdt', '_T'])
srs = srs.merge(srs2, on=['Day'])
# ==================================== Fit models ====================================


# Some more Data processing
# Take absolute values
data_perf['RATE_P0'] = data_perf.RATE_P0.abs()
data_perf['dTdx'] = data_perf.dTdx.abs()
data_perf['rate_log'] = np.log(data_perf.RATE_P0)
# to_drop = (data_perf.Day> 0.0033) & (data_perf.Day<=0.52)
# single_data = data_perf.loc[(data.IB==1225) & ~to_drop & (data.RATE_P0>-12)].dropna()
keep = (data_perf.Day> 0.0033) & (data.IB==1031)
single_data = data_perf.loc[keep].dropna()

# single_data['logT'] = np.log(single_data.T_noisy)
X_train = single_data[['dTdt', 'dTdx', 'T']]
y_train = single_data['RATE_P0'].abs()
y_log_train = single_data.rate_log
nm = Normalizer()
stdz = StandardScaler()
#X_norm_train = nm.fit_transform(X_train)
#X_std_train = stdz.fit_transform(X_train)

# X_norm_test = nm.transform(X_train)
# X_std_test = stdz.transform(X_train)
# X_poly2_test = poly2.transform(X_norm_test)
# X_poly3_test = poly3.transform(X_norm_test)
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(X_norm_train)



# -----------------
# Drop first steps of simulation (noise)
keep_test = (data_perf.Day> 0.0033) & (data.IB==1031)
single_data = data_perf.loc[keep].dropna()

X_train = single_data[['dTdt', 'dTdx', 'T']]
y_train = single_data['RATE_P0'].abs()
y_log_train = single_data.rate_log

# Train the lasso
lasso_cv_model3 = LassoCV(n_alphas=100, cv=10, fit_intercept=False, n_jobs=-1).fit(X_poly3, y_log_train)
lasso_model3 = Lasso(alpha=lasso_cv_model3.alpha_, fit_intercept=False).fit(X_poly3, y_log_train)
pred_cv_lasso_poly3 = np.exp(cross_val_predict(lasso_model3, X_poly3, y_log_train, cv=10))

# Tree
forest = RandomForestRegressor(n_jobs=-1, oob_score=True)
param_grid = {'n_estimators': [10,20,50,100,250,500],
              'max_features': ['auto', 'sqrt']}
CV_tree_poly3 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error',
                             n_jobs=-1).fit(X_poly3, y_train)

best_tree_3 = CV_tree_poly3.best_estimator_
pred_cv_tree = cross_val_predict(best_tree_3, X_poly3, y_train, cv=10)

preds = pd.DataFrame({'y': y_train,
                         'lasso_poly3': pred_cv_lasso_poly3,
                        'forest': pred_cv_tree})
