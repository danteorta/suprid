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
data = pd.read_pickle(folder + 'wellbore_data_3days_300_rate')
data['Day'] = data.Day.astype(float)

# Add noise
max_noise = 0.5
noise = (np.random.random(len(data))-0.5) * max_noise
data['T_noisy'] = data['T'] + noise

# Pivot to time-series
grouping = data.groupby(['Day', 'x']).T.mean().reset_index()
t_series = grouping.pivot(values='T', index='x', columns='Day')

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

# data_perf = data.loc[data.IB.isin([4374, 2737, 2472, 1031, 1225]),:]
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
from sklearn.linear_model import LinearRegression, LassoCV, Lasso
from sklearn.preprocessing import Normalizer, StandardScaler, PolynomialFeatures
from sklearn.model_selection import cross_val_predict, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Data processing
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
X_norm_train = nm.fit_transform(X_train)
X_std_train = stdz.fit_transform(X_train)

# X_norm_test = nm.transform(X_train)
# X_std_test = stdz.transform(X_train)
# X_poly2_test = poly2.transform(X_norm_test)
# X_poly3_test = poly3.transform(X_norm_test)


poly2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly2 = poly2.fit_transform(X_norm_train)
poly3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly3 = poly3.fit_transform(X_norm_train)



# -----------------
keep_test = (data_perf.Day> 0.0033) & (data.IB==1031)
single_data = data_perf.loc[keep].dropna()

# single_data['logT'] = np.log(single_data.T_noisy)
X_train = single_data[['dTdt', 'dTdx', 'T']]
y_train = single_data['RATE_P0'].abs()
y_log_train = single_data.rate_log
# ----------------------


# fIT A LINEAR REGRESSION
lin_reg = LinearRegression()
# lin_reg.fit(X_std,y)


# CV
pred_cv_ols = np.exp(cross_val_predict(lin_reg, X_norm_train, y_log_train, cv=10))
pred_cv_ols_poly2 = np.exp(cross_val_predict(lin_reg, X_poly2, y_log_train, cv=10))
pred_cv_ols_poly3 = np.exp(cross_val_predict(lin_reg, X_poly3, y_log_train, cv=10))

# Train the lasso
lasso_cv_model = LassoCV(n_alphas=100, cv=10, fit_intercept=False, n_jobs=-1).fit(X_std_train, y_log_train)
lasso_cv_model2 = LassoCV(n_alphas=100, cv=10, fit_intercept=False, n_jobs=-1).fit(X_poly2, y_log_train)
lasso_cv_model3 = LassoCV(n_alphas=100, cv=10, fit_intercept=False, n_jobs=-1).fit(X_poly3, y_log_train)
lasso_model = Lasso(alpha=lasso_cv_model.alpha_, fit_intercept=False).fit(X_std_train, y_log_train)
lasso_model2 = Lasso(alpha=lasso_cv_model2.alpha_, fit_intercept=False).fit(X_poly2, y_log_train)
lasso_model3 = Lasso(alpha=lasso_cv_model3.alpha_, fit_intercept=False).fit(X_poly3, y_log_train)
pred_cv_lasso = np.exp(cross_val_predict(lasso_model, X_std_train, y_log_train, cv=10))
pred_cv_lasso_poly2 = np.exp(cross_val_predict(lasso_model2, X_poly2, y_log_train, cv=10))
pred_cv_lasso_poly3 = np.exp(cross_val_predict(lasso_model3, X_poly3, y_log_train, cv=10))

# Tree
forest = RandomForestRegressor(n_jobs=-1, oob_score=True)
param_grid = {'n_estimators': [10,20,50,100,250,500],
              'max_features': ['auto', 'sqrt']}
CV_tree = GridSearchCV(estimator=forest, param_grid=param_grid, scoring='neg_mean_squared_error',
                       cv=10, n_jobs=-1).fit(X_train, y_train)
CV_tree_poly2 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error',
                             n_jobs=-1).fit(X_poly2, y_train)
CV_tree_poly3 = GridSearchCV(estimator=forest, param_grid=param_grid, cv=10, scoring='neg_mean_squared_error',
                             n_jobs=-1).fit(X_poly3, y_train)

best_tree_3 = CV_tree_poly3.best_estimator_
pred_cv_tree = cross_val_predict(best_tree_3, X_poly3, y_train, cv=10)

pred_tree_3 = pd.DataFrame( best_tree_3.predict(X_poly3))
# pred_tree_test = best_tree_3.predict(X_poly3_test)
#1
#
#
# # -----------------
# svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1), cv=10,
#                    param_grid={"C": [1e0, 1e1, 1e2, 1e3],
#                                "gamma": np.logspace(-5, 2, 10)})
# svr.fit(X_std_train, y_train)
# svm_model = SVR(kernel='poly', **svr.best_params_)
# pred_cv_svm = cross_val_predict(svm_model, X_norm_train, y_train, cv=10)
# svm_model.fit(X_norm_train, y_train)
# -----------------
cv_preds = pd.DataFrame({'y': y_train,
                         'ols': pred_cv_ols,
                         'ols_poly2': pred_cv_ols_poly2,
                         'ols_poly3': pred_cv_ols_poly3,
                         'lasso': pred_cv_lasso,
                         'lasso_poly2': pred_cv_lasso_poly2,
                         'lasso_poly3': pred_cv_lasso_poly3})

test_preds = pd.DataFrame({'y': y_train,
                         'lasso_poly2': np.exp(lasso_model2.predict(X_poly2_test)),
                         'lasso_poly3': np.exp(lasso_model3.predict(X_poly3_test))})



ols_scores = {'ols': {'r2': r2_score(y_log_train, pred_cv_ols),
                      'mae': mean_absolute_error(y_log_train, pred_cv_ols),
                      'mse': mean_squared_error(y_log_train, pred_cv_ols)},
              'ols_poly2': {'r2': r2_score(y_log_train, pred_cv_ols_poly2),
                             'mae': mean_absolute_error(y_log_train, pred_cv_ols_poly2),
                             'mse': mean_squared_error(y_log_train, pred_cv_ols_poly2)},
              'ols_poly3': {'r2': r2_score(y_log_train, pred_cv_ols_poly3),
                             'mae': mean_absolute_error(y_log_train, pred_cv_ols_poly3),
                             'mse': mean_squared_error(y_log_train, pred_cv_ols_poly3)},
              'lasso_2': {'r2': r2_score(y_log_train, pred_cv_lasso_poly2),
                            'mae': mean_absolute_error(y_log_train, pred_cv_lasso_poly2),
                            'mse': mean_squared_error(y_log_train, pred_cv_lasso_poly2)},
              'lasso_3': {'r2': r2_score(y_log_train, pred_cv_lasso_poly2),
                          'mae': mean_absolute_error(y_log_train, pred_cv_lasso_poly3),
                          'mse': mean_squared_error(y_log_train, pred_cv_lasso_poly3)},
              'forest': {'r2': r2_score(y_log_train, pred_cv_lasso_poly2),
                          'mae': mean_absolute_error(y_train, pred_tree_3),
                          'mse': mean_squared_error(y_train, pred_cv_tree)}
               }



# Plotting
plt.scatter(x=y_log_train.values, y= pred_cv_ols, label='ols')
plt.scatter(x=y_log_train.values, y= pred_cv_ols_poly2, label='ols_poly2')
plt.scatter(x=y_log_train.values, y= pred_cv_ols_poly3, label='ols_poly3')
plt.scatter(x=y_log_train.values, y=pred_cv_lasso_poly2, label='lasso_poly2')
plt.plot([y_log_train.min(), y_log_train.max()], [y_log_train.min(), y_log_train.max()])
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
