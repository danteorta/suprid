import pandas as pn
from matplotlib import pyplot as pl
from sklearn.cluster import KMeans
from sklearn import tree
from sklearn.linear_model import LinearRegression

def add_normalized_cols(my_dataframe):
    """
    Function to add columns for Machine-Learning model
    :param my_dataframe:
    :return: nothing
    """
    my_dataframe['time_norm'] = my_dataframe['time'].astype(float) /\
                                 (my_dataframe['time'].max() - my_dataframe['time'].min())
    my_dataframe['temp_norm'] = (my_dataframe['T'] - my_dataframe['T'].min()) / \
                                 (my_dataframe['T'].max() - my_dataframe['T'].min())
    my_dataframe['len_norm'] = (my_dataframe.IB.astype(float) - my_dataframe.IB.min()) / \
                               (my_dataframe.IB.max() - my_dataframe.IB.min())
    my_dataframe['fracture'] = 0
    my_dataframe.loc[my_dataframe.IB.isin([320, 5654, 5656]), 'fracture'] = 1

    my_dataframe.loc[:, 'time_deriv'] = (my_dataframe.groupby(['IB', 'time'])['T'].mean() -
                                   my_dataframe.groupby(['IB', 'time'])['T'].mean().shift(-1)).values
    my_dataframe.loc[:, 'space_deriv'] = (my_dataframe.groupby(['time', 'IB'])['T'].mean() -
                                   my_dataframe.groupby(['time', 'IB'])['T'].mean().shift(-1)).values
    my_dataframe.dropna(how='any', subset=['time_deriv','space_deriv'], inplace=True)

def train_tree(train_df, train_cols):
    X = train_df.loc[:, train_cols].values
    Y = train_df.loc[:, 'fracture']
    clf_1 = tree.DecisionTreeClassifier()
    clf_1 = clf_1.fit(X, Y)
    return clf_1

# --------------------------- Define a few paths ---------------------------
it_name = 'DANTEMODEL_T'
# model_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/test_run/gprs_repo/'
model_path = 'D:/Users/dorta/Dropbox/Stanford/Research/workspace/sandbox/80/'
rates_path = model_path + '{0}.rates.txt'.format(it_name)
vars_path = model_path + '{0}.vars.txt'.\
    format(it_name)
fips_path = model_path + '{0}.FIPS.txt'.format(it_name)
wells_path = model_path + '{0}.WELLS.StdWell_W.txt'.\
    format(it_name)
trans_path = model_path + 'model\\trans.txt'

# import training data
train_data = pn.read_pickle('D:/Users/dorta/Dropbox/Stanford/Research/workspace/sandbox/90/wellbore_data')
test_1 = pn.read_pickle('D:/Users/dorta/Dropbox/Stanford/Research/workspace/sandbox/85/wellbore_data')
test_2 = pn.read_pickle('D:/Users/dorta/Dropbox/Stanford/Research/workspace/sandbox/80/wellbore_data')


# ----------------- Preprocessing -----------------
# Add some normalized columns
add_normalized_cols(train_data)
add_normalized_cols(test_1)
add_normalized_cols(test_2)

# Train Trees
tcols_1 = ['temp_norm']
tcols_2 = ['temp_norm', 'time_norm']
tcols_3 = ['space_deriv']
tcols_4 = ['temp_norm', 'space_deriv']
tcols_5 = ['temp_norm', 'space_deriv', 'time_deriv']

tree_1 = train_tree(train_data, tcols_1)
tree_2 = train_tree(train_data, tcols_2)
tree_3 = train_tree(train_data, tcols_3)
tree_4 = train_tree(train_data, tcols_4)
tree_5 = train_tree(train_data, tcols_5)

regs ={}
coefs = []
for nn in range(len(tcols)):
    my_model = LinearRegression(fit_intercept=False)
    my_model.fit(train_data[tcols[nn]], train_data.loc[:, 'fracture'])
    regs['{0}'.format(nn)] = my_model
    coefs.append({tcols[nn][x]: my_model.coef_[x] for x in range(len(tcols[nn]))})

# Compute feature importances
tcols = [tcols_1, tcols_2, tcols_3, tcols_4, tcols_5]
trees = [tree_1, tree_2, tree_3, tree_4, tree_5]
fts = [{tcols[y][x]: trees[y].feature_importances_[x] for x in range(len(tcols[y]))} for y in range(len(tcols))]
feature_importances = pn.DataFrame(fts)

X = train_data[tcols_4].values
import numpy as np
from matplotlib import pyplot as plt
from itertools import product

clf1 = tree_4

x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1],
                        ['Decision Tree (depth=4)']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    axarr[idx[0], idx[1]].set_title(tt)







# Test
test_1['fracture_pred_1'] = tree_1.predict(test_1.loc[:, tcols_1])
test_1['fracture_pred_2'] = tree_2.predict(test_1.loc[:, tcols_2])
test_1['fracture_pred_3'] = tree_3.predict(test_1.loc[:, tcols_3])
test_1['fracture_pred_4'] = tree_4.predict(test_1.loc[:, tcols_4])
test_1['fracture_pred_5'] = tree_5.predict(test_1.loc[:, tcols_5])

misclass_rate = []
for n in range(1 ,len(tcols)+1):
    mr = (test_1['fracture_pred_{0}'.format(n)] - test_1.fracture).abs().mean()
    misclass_rate.append(mr)


test_2['fracture_pred_1'] = tree_1.predict(test_2.loc[:, tcols_1])
test_2['fracture_pred_2'] = tree_2.predict(test_2.loc[:, tcols_2])
test_2['fracture_pred_3'] = tree_3.predict(test_2.loc[:, tcols_3])


ts_1_1 = test_1.pivot(values='fracture_pred_1', index='IB', columns='time')
ts_1_2 = test_1.pivot(values='fracture_pred_2', index='IB', columns='time')
ts_1_3 = test_1.pivot(values='fracture_pred_3', index='IB', columns='time')


ts_2_1 = test_2.pivot(values='fracture_pred_1', index='IB', columns='time')
ts_2_2 = test_2.pivot(values='fracture_pred_2', index='IB', columns='time')
ts_2_3 = test_2.pivot(values='fracture_pred_3', index='IB', columns='time')


pl.pcolor(ts_2_3.values)
pl.colorbar()
pl.show(False)

test_1.loc[:, 'time_deriv'] = (test_1.groupby(['IB', 'time'])['T'].mean() -
                               test_1.groupby(['IB', 'time'])['T'].mean().shift(-1)).values
test_2.loc[:, 'time_deriv'] = (test_2.groupby(['IB', 'time'])['T'].mean() -
                               test_2.groupby(['IB', 'time'])['T'].mean().shift(-1)).values


train_data.groupby(['IB','time']).apply(shiftCol, newCol='s_t', col='T')

def shiftCol(grp, newCol, col):
    grp[newCol] = grp[col].shift()
    return grp

def shift_subst(x, col):
    return x.loc[:, col] - x.loc[:, col].shift(-1)


# ---- CASE 2 - train with Temperature ----
# Train

# Test
X_test_1 = test_1.loc[:, train_cols].values
X_test_2 = test_2.loc[:, train_cols].values

test_1['fracture_pred'] = clf_1.predict(X_test_1)
test_2['fracture_pred'] = clf_1.predict(X_test_2)





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

