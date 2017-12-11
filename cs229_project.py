import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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


# Import data
folder = 'C:/Users/dorta/Dropbox/Stanford/CS 229/project/'
data = pd.read_pickle(folder + 'wellbore_data_half_two_prod')
data['Day'] = data.Day.astype(float)

# Add noise
max_noise = 0.5
noise = (np.random.random(len(data))-0.5) * max_noise
data['T_noisy'] = data['T'] + noise

# Pivot to time-series
grouping = data.groupby(['Day', 'x']).T_noisy.mean().reset_index()
t_series = grouping.pivot(values='T_noisy', index='x', columns='Day')

# Plot some examples of tseries
# t_series[[0.028, 0.5, 0.9014, 1.5572, 2.0525, 2.5]].plot(legend=True)
# plt.show(False)

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

data_perf = data.loc[data.IB.isin([4374, 2737, 2472, 1031, 1225]),:]
data_perf = data_perf[['IB','p','T','TS','Day','poro','x','y','z','RATE_P0','T_noisy', 'dTdt', 'dTdx']]

# -------------
rate_series = tserialize(data, ['x','Day'],'RATE_P0')
rate = data_perf.pivot_table(values='RATE_P0',index='Day',columns='IB').reset_index()
dTdx = data_perf.pivot_table(values='dTdx',index='Day',columns='IB').reset_index()
dTdt = data_perf.pivot_table(values='dTdt',index='Day',columns='IB').reset_index()
T = data_perf.pivot_table(values='T_noisy',index='Day',columns='IB').reset_index()

srs = rate.merge(dTdx, on=['Day'],suffixes=['_rate','_dTdx'])
srs2 = dTdt.merge(T, on=['Day'], suffixes=['_dTdt', '_T'])
srs = srs.merge(srs2, on=['Day'])
# /////////////fit model
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.preprocessing import Normalizer, StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR

# Data processing
to_drop = (data.Day> 0.5) & (data.Day<=0.52)
single_data = data_perf.loc[(data.IB==1225) & ~to_drop & (data.RATE_P0>-12)].dropna()
# single_data['logT'] = np.log(single_data.T_noisy)
X = single_data[['dTdt', 'dTdx', 'T']]
y = single_data['RATE_P0'].abs()
y_log = np.log(np.abs(single_data['RATE_P0']))
nm = Normalizer()
stdz = StandardScaler()
X_norm = nm.fit_transform(X)
X_std = stdz.fit_transform(X)

poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(X_norm)
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(X_norm)

# fIT A LINEAR REGRESSION
lin_reg = LinearRegression()
# lin_reg.fit(X_std,y)


# CV
pred_cv_ols = np.exp(cross_val_predict(lin_reg, X_norm, y_log, cv=10))
pred_cv_ols_poly2 = np.exp(cross_val_predict(lin_reg, X_poly2, y_log, cv=10))
pred_cv_ols_poly3 = np.exp(cross_val_predict(lin_reg, X_poly3, y_log, cv=10))

# Train the lasso
lasso_cv_model = LassoCV(n_alphas=100, cv=10, fit_intercept=False, n_jobs=-1).fit(X_poly2, y_log)
lasso_cv_model3 = LassoCV(n_alphas=100, cv=10, fit_intercept=False, n_jobs=-1).fit(X_poly3, y_log)
lasso_model = Lasso(alpha=lasso_cv_model.alpha_, fit_intercept=False)
lasso_model3 = Lasso(alpha=lasso_cv_model3.alpha_, fit_intercept=False)
pred_cv_lasso_poly2 = np.exp(cross_val_predict(lasso_model, X_poly2, y_log, cv=10))
pred_cv_lasso_poly3 = np.exp(cross_val_predict(lasso_model3, X_poly3, y_log, cv=10))

svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-5, 2, 10)})
svr.fit(X_std, y)

svm_model = SVR(kernel='rbf', gamma=0.1)
pred_cv_svm = cross_val_predict(svm_model, X_norm, y, cv=10)

ols_scores = {'ols': {'r2': r2_score(y_log, pred_cv_ols),
                      'mae': mean_absolute_error(y_log, pred_cv_ols),
                      'mse': mean_squared_error(y_log, pred_cv_ols)},
              'ols_poly2': {'r2': r2_score(y_log, pred_cv_ols_poly2),
                             'mae': mean_absolute_error(y_log, pred_cv_ols_poly2),
                             'mse': mean_squared_error(y_log, pred_cv_ols_poly2)},
              'ols_poly3': {'r2': r2_score(y_log, pred_cv_ols_poly3),
                             'mae': mean_absolute_error(y_log, pred_cv_ols_poly3),
                             'mse': mean_squared_error(y_log, pred_cv_ols_poly3)},
              'lasso_2': {'r2': r2_score(y_log, pred_cv_lasso_poly2),
                            'mae': mean_absolute_error(y_log, pred_cv_lasso_poly2),
                            'mse': mean_squared_error(y_log, pred_cv_lasso_poly2)},
              'lasso_3': {'r2': r2_score(y_log, pred_cv_lasso_poly2),
                          'mae': mean_absolute_error(y_log, pred_cv_lasso_poly2),
                          'mse': mean_squared_error(y_log, pred_cv_lasso_poly2)},
               }

cv_preds = pd.DataFrame({'y': y,
                         'ols': pred_cv_ols,
                         'ols_poly2': pred_cv_ols_poly2,
                         'ols_poly3': pred_cv_ols_poly3,
                         'lasso_poly2': pred_cv_lasso_poly2,
                         'lasso_poly3': pred_cv_lasso_poly3,
                         'svm': pred_cv_svm})


# Plotting
plt.scatter(x=y_log.values, y= pred_cv_ols, label='ols')
plt.scatter(x=y_log.values, y= pred_cv_ols_poly2, label='ols_poly2')
plt.scatter(x=y_log.values, y= pred_cv_ols_poly3, label='ols_poly3')
plt.scatter(x=y_log.values, y=pred_cv_lasso_poly2, label='lasso_poly2')
plt.plot([y_log.min(), y_log.max()], [y_log.min(), y_log.max()])
plt.legend()
plt.show(False)

# # Display results
# m_log_alphas = -np.log10(lasso_model.alphas_)
#
# plt.figure()
# ymin, ymax = y.min(), y.max()
# plt.plot(m_log_alphas, lasso_model.mse_path_, ':')
# plt.plot(m_log_alphas, lasso_model.mse_path_.mean(axis=-1), 'k',
#          label='Average across the folds', linewidth=2)
# plt.axvline(-np.log10(lasso_model.alpha_), linestyle='--', color='k',
#             label='alpha: CV estimate')
#
# plt.legend()
# plt.xlabel('-log(alpha)')
# plt.ylabel('Mean square error')
# plt.axis('tight')
# plt.ylim(ymin, ymax)
