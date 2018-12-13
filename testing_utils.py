import pandas as pd
import math
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingRegressor
import matplotlib.pyplot as plt


def rmsle(y, y_pred):
    """
    return root mean squared error for set of true labels
    and set of predictions
    """

    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def calculate_error(train_true,
                    train_pred,
                    test_true,
                    test_pred,
                    outcome_var):
    if outcome_var == 'LogSalePrice':
        test_rmse = rmsle(np.exp(list(test_true)), np.exp(test_pred))
        train_rmse = rmsle(np.exp(list(train_true)), np.exp(train_pred))
    else:
        test_rmse = rmsle(list(test_true), test_pred)
        train_rmse = rmsle(list(train_true), train_pred)
    return test_rmse, train_rmse

# ensemble_to_use should be a df row with the following og_columns
# it's possible you could use it with a non-ensemble - i'm not actually sure
# if it would need adjustments or not
def retrain_on_all_data_and_get_final_preds(ensemble_to_use,
                                            train_data,
                                            test_data):
    indiv_model_preds = []

    for i, model in enumerate(ensemble_to_use['Models'].values[0]):
        features_to_use = ensemble_to_use['Features'].values[0][i]
        outcome_var_to_use = ensemble_to_use['Outcome_Vars'].values[0][i]

        # fit the model on all training data this time
        model.fit(train_data[features_to_use],
                  train_data[outcome_var_to_use])

        # re-predict on test data
        init_preds = model.predict(test_data[features_to_use])
        if outcome_var_to_use == 'LogSalePrice':
            init_preds = np.exp(init_preds)

        indiv_model_preds.append(init_preds)

    # do the averaging
    final_preds = []
    for i in range(len(indiv_model_preds[0])):
        final_preds.append(sum([preds_list[i] for preds_list in indiv_model_preds]) / 
                           len(indiv_model_preds))

    return final_preds

def make_kaggle_submission(predictions, test_df, submission_name):
    ids = [int(id) for id in test_df['Id'].values]
    submission_df = pd.DataFrame(data={'Id': ids, 'SalePrice': predictions})
    submission_df.to_csv(submission_name, index = False)

def plot_error_against_var(model, outcome_var, feature_set, plot_features, dev_df):

    # Calculate predictions for each observation
    pred_LR = pd.DataFrame(model.predict(dev_df[feature_set]), columns = ['pred'])

    # Calculate error for each observation
    dev_df2 = pd.concat([dev_df.reset_index(drop=True), pred_LR], axis = 1)
    dev_df2['error'] = dev_df2['pred'] - dev_df2[outcome_var]

    # Plot Errors:
    col_count = 3
    row_count = len(plot_features)//col_count + 1

    fig, ax = plt.subplots(row_count, col_count, figsize = (20,10))

    for counter, var in enumerate(plot_features):
        col_position = counter%col_count
        row_position = counter//col_count

        ax[row_position, col_position].scatter(dev_df2[var], dev_df2['error'])
        ax[row_position, col_position].axhline(y=0)
        ax[row_position, col_position].set_xlabel(var)
        ax[row_position, col_position].set_ylabel('error')

    fig.suptitle(model)
    fig.subplots_adjust(hspace=0.5)
    plt.show()


def create_error_correlation_table(model,
                                   outcome_var,
                                   feature_set,
                                   dev_df):
    
    '''
    finds correlation between absolute value of error
    and each feature
    '''
    
    final_data = {'col': feature_set}
    dev_df = dev_df.reset_index()
    
    dev_preds = model.predict(dev_df[feature_set])

    rmsles = []
    for i in range(len(dev_preds)):
        rmsles.append(testing.rmsle([dev_df[outcome_var][i]], [dev_preds[i]]))

    plt.clf()
    plt.hist(rmsles, bins=20)
    plt.xlabel("RMSLE")
    plt.ylabel("Number of Occurrences")
    plt.show()

    dev_df['linear_reg_errors'] = rmsles

    corrs = []
    for col in feature_set:
        try:
            cor = np.corrcoef(abs(dev_df['linear_reg_errors']), dev_df[col])[0,1]
            corrs.append(cor)
        except:
            pass

    final_data['correlation'] = corrs 
    
    corrs_df = pd.DataFrame(data=final_data)
    corrs_df = corrs_df.dropna()
    return corrs_df


########################
# Stacking Utility Functions
########################

# Load Kaggle test set
def get_best_specs_from_each_family(family_df):
    model = family_df['Model'].iloc[0][0]
    return(model)

def change_to_saleprice(predictions):
    if np.mean(predictions) < 1000:
        return(np.exp(predictions))
    else:
        return(predictions)
    
def stack_ensemble(train_data, dev_data, model1, model2, model3, kaggle_flag):
    # Drop these columns in train
    dropcols_train = ['YrMoSold', 'SalePrice', 'LogSalePrice']
    
    # Assign workaround if kaggle_flag == True:
    if kaggle_flag == True:
        dropcols_dev = ['YrMoSold']
    else:
        dropcols_dev = ['YrMoSold', 'SalePrice', 'LogSalePrice', 'baseline_pred']

    # Get boosted predictions
    best_boosted_model = model1
    boosted_pred = best_boosted_model.predict(train_data.drop(columns=dropcols_train)) # how to include the cross validation here
    boosted_pred = change_to_saleprice(boosted_pred)
    
    boosted_test_pred = best_boosted_model.predict(dev_data.drop(columns=dropcols_dev))
    boosted_test_pred = change_to_saleprice(boosted_test_pred)
    
    # Get LR predictions
    best_LR_model = model2
    LR_pred = best_LR_model.predict(train_data.drop(columns=dropcols_train))
    LR_pred = change_to_saleprice(LR_pred)
    
    LR_test_pred = best_LR_model.predict(dev_data.drop(columns=dropcols_dev))
    LR_test_pred = change_to_saleprice(LR_test_pred)

    # Get bagged predictions
    best_bagg_model = model3
    bagg_pred = best_bagg_model.predict(train_data.drop(columns=dropcols_train))
    bagg_pred = change_to_saleprice(bagg_pred)

    bagg_test_pred = best_bagg_model.predict(dev_data.drop(columns=dropcols_dev))
    bagg_test_pred = change_to_saleprice(bagg_test_pred)

    # Create stacked model
    trainpred = pd.DataFrame(np.column_stack((boosted_pred, LR_pred, bagg_pred)), columns = ['boosted', 'LR', 'bagg'])
    testpred  = pd.DataFrame(np.column_stack((boosted_test_pred, LR_test_pred, bagg_test_pred)), columns = ['boosted', 'LR', 'bagg'])

    # Fit the ensemble parameters to fit the all the data
    ensemble_LR = LinearRegression()
    ensemble_LR.fit(trainpred, train_data['SalePrice'])
    finaltrainpred = ensemble_LR.predict(trainpred)

    # Fit the final predictions
    finaltestpred = ensemble_LR.predict(testpred)

    # Calculate the errgor (note this is the whole dataset so test/train are the same if you choose kaggle_flag == True)
    if kaggle_flag == True:
        error = calculate_error(train_true = np.array(train_data['SalePrice']), 
                                train_pred = np.array(finaltrainpred),
                                test_true  = np.array(train_data['SalePrice']),
                                test_pred  = np.array(finaltrainpred),
                                outcome_var = 'SalePrice')
    else:
        error = calculate_error(train_true = np.array(train_data['SalePrice']), 
                        train_pred = np.array(finaltrainpred),
                        test_true  = np.array(dev_data['SalePrice']),
                        test_pred  = np.array(finaltestpred),
                        outcome_var = 'SalePrice')


#     print(error)
#     print(np.mean(boosted_test_pred), np.mean(LR_test_pred), np.mean(bagg_test_pred))
#     print(np.mean(finaltestpred))
#     print(finaltestpred.shape)
    
    return [finaltestpred, testpred]


def plot_error_against_var(model, outcome_var, feature_set, plot_features, dev_df):
    
    # Calculate predictions for each observation
    pred_LR = pd.DataFrame(model.predict(dev_df[feature_set]), columns = ['pred'])
    
    # Calculate error for each observation
    dev_df2 = pd.concat([dev_df.reset_index(drop=True), pred_LR], axis = 1)
    dev_df2['error'] = dev_df2['pred'] - dev_df2[outcome_var]
    
    # Plot Errors:
    col_count = 3
    row_count = len(plot_features)//col_count + 1

    fig, ax = plt.subplots(row_count, col_count, figsize = (20,10))

    for counter, var in enumerate(plot_features): 
        col_position = counter%col_count
        row_position = counter//col_count

        ax[row_position, col_position].scatter(dev_df2[var], dev_df2['error'])
        ax[row_position, col_position].axhline(y=0)
        ax[row_position, col_position].set_xlabel(var)
        ax[row_position, col_position].set_ylabel('error')
     
    fig.suptitle(model)
    fig.subplots_adjust(hspace=1.5)
    plt.show()