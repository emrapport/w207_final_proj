import pandas as pd
import math
import numpy as np
import logging


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
