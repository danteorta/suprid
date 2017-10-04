import pandas as pn
from matplotlib import pyplot as pl
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

# Import the training data
train_data = pn.read_pickle('D:/Users/dorta/Dropbox/Stanford/Research/workspace/sandbox/90/wellbore_data')
test_1 = pn.read_pickle('D:/Users/dorta/Dropbox/Stanford/Research/workspace/sandbox/85/wellbore_data')
test_2 = pn.read_pickle('D:/Users/dorta/Dropbox/Stanford/Research/workspace/sandbox/80/wellbore_data')


# ----------------- Preprocessing -----------------
# Add some normalized columns
add_normalized_cols(train_data)
add_normalized_cols(test_1)
add_normalized_cols(test_2)

# Train Trees
# Define cases -- Diferent training columns
tcols = {'1': ['temp_norm'],
         '2': ['temp_norm', 'time_norm'],
         '3': ['space_deriv'],
         '4': ['temp_norm', 'space_deriv'],
         '5': ['temp_norm', 'space_deriv', 'time_deriv'],
         '6': ['time_deriv']}

# Train N different trees
trees = {case: train_tree(train_data, tcols[case]) for case in tcols.keys()}
# Compute feature importances for each tree
fts = {ky: {tcols[ky][x]: trees[ky].feature_importances_[x] for x in range(len(tcols[ky]))} for ky in tcols.keys()}
feature_importances = pn.DataFrame(fts)

# Try some linear regressions
regs = {}
coefs = []
for nn in range(len(tcols)):
    my_model = LinearRegression(fit_intercept=False)
    my_model.fit(train_data[tcols[nn]], train_data.loc[:, 'fracture'])
    regs['{0}'.format(nn)] = my_model
    coefs.append({tcols[nn][x]: my_model.coef_[x] for x in range(len(tcols[nn]))})


# Make predictions in the test dataset for all trees
misclass_rate = []
for case in tcols.keys():
    # Test dataset 1
    x_test_1 = test_1.loc[:, tcols[case]]
    test_1['fracture_pred_{}'.format(case)] = trees[case].predict(x_test_1)
    # Compute the misclassification rate
    mr = (test_1['fracture_pred_{}'.format(case)] - test_1.fracture).abs().mean()
    misclass_rate.append(mr)

    # Test dataset 2 -- Kinda irrelevant at this point
    x_test_2 = test_2.loc[:, tcols[case]]
    test_2['fracture_pred_{}'.format(case)] = trees[case].predict(x_test_2)



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

