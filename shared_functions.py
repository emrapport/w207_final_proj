import pandas as pd
import math
import numpy as np
import logging

def train_and_test_cross_validated(cross_val_lists,
                                   model,
                                   param_set,
                                   outcome_var,
                                   features):
    all_test_rmsles = []
    all_train_rmsles = []
    all_fit_models = []
    for dictionary in cross_val_lists:
        model_instance = model(**param_set)
        (test_rmse,
        train_rmse,
        fit_model) = train_and_test(dictionary['train_df'],
                                    dictionary['dev_df'],
                                    model_instance,
                                    outcome_var,
                                    features)
        all_test_rmsles.append(test_rmse)
        all_train_rmsles.append(train_rmse)
        all_fit_models.append(fit_model)
    #logging.debug("All test RMSLES: {}".format(all_test_rmsles))
    return np.mean(all_test_rmsles), np.mean(all_train_rmsles), all_fit_models

def predict_via_vote_cross_validated(cross_val_lists,
                                     model_list,
                                     features_list,
                                     outcome_var):
    all_test_rmsles = []

    for dictionary in cross_val_lists:
        new_df = predict_via_vote(model_list,
                                  features_list,
                                  dictionary['dev_df'])
        # the ones are a hack to make it not error for train RMSLES
        # since train doesn't matter in this voting world
        test_rmsle = calculate_error([1],
                                     [1],
                                     new_df[outcome_var],
                                     new_df['vote_by_avg'],
                                     outcome_var)[0]
        all_test_rmsles.append(test_rmsle)

    #logging.debug("All test RMSLES: {}".format(all_test_rmsles))
    return np.mean(all_test_rmsles)

def predict_via_vote(model_list, feature_set_list, dev_df):

    new_dev_df = dev_df.reset_index()

    cols_to_avg = []
    for i, model in enumerate(model_list):
        cols_to_avg.append(i)
        dev_preds = model.predict(new_dev_df[feature_set_list[i]])
        new_dev_df[i] = dev_preds

    new_dev_df['vote_by_avg'] = (sum([new_dev_df[i] for i in cols_to_avg])
                                / len(cols_to_avg))
    #logging.debug("new dev df vote by avg: {}".format(new_dev_df['vote_by_avg']))
    return new_dev_df

def rmsle(y, y_pred):
    """
    return root mean squared error for set of true labels
    and set of predictions
    """

    assert len(y) == len(y_pred)
    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

def train_and_test(train_df,
                   dev_df,
                   model,
                   outcome_var,
                   features):
    """
    Args:
    train_df: df of training rows. must have all columns in features
    dev_df: df of dev rows. must have all columns in features
    model: scikit-learn model, already initialized with desired hparams
    outcome_var: either 'SalePrice' or 'LogSalePrice'
    features: a list of features to use for regression

    Returns:
    rmse: float, the rmse for the model on the dev set

    """
    fit_model = model.fit(train_df[features], train_df[outcome_var])
    train_preds = fit_model.predict(train_df[features])
    dev_preds = fit_model.predict(dev_df[features])

    test_rmse, train_rmse = calculate_error(train_df[outcome_var],
                                            train_preds,
                                            dev_df[outcome_var],
                                            dev_preds,
                                            outcome_var)

    return test_rmse, train_rmse, fit_model

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

def try_different_models(cross_val_list,
                         models_to_params_list,
                         outcome_vars,
                         feature_sets):
    """
    Args:
    train_df: df of training rows. must have all columns in features
    dev_df: df of dev rows. must have all columns in features
    models: list of scikit-learn modesl, already initialized with desired hparams
    outcome_vars: list of outcome var strings
    features_sets: list of lists of features to try

    Returns:
    df: a dataframe of results for all models

    """
    models_tried = []
    params_tried = []
    outcome_vars_tried = []
    num_features_tried = []
    features_tried = []
    train_rmses_tried = []
    test_rmses_tried = []

    for model, param_sets in models_to_params_list.items():
        for outcome_var in outcome_vars:
            for param_set in param_sets:
                for feature_set in feature_sets:

                    try:
                        (test_rmse,
                        train_rmse,
                        fit_models) = train_and_test_cross_validated(cross_val_list,
                                                              model,
                                                              param_set,
                                                              outcome_var,
                                                              feature_set)

                        models_tried.append(fit_models)
                        params_tried.append(param_set)
                        outcome_vars_tried.append(outcome_var)
                        features_tried.append(feature_set)
                        num_features_tried.append(len(feature_set))
                        train_rmses_tried.append(train_rmse)
                        test_rmses_tried.append(test_rmse)

                    except Exception as e:
                        logging.debug(e)

    scores_df = pd.DataFrame(data={'Model': models_tried,
                                   'Params': params_tried,
                                   'Outcome Var': outcome_vars_tried,
                                   'Features': features_tried,
                                   'Num Features': num_features_tried,
                                   'Train MSE': train_rmses_tried,
                                   'Root MSE': test_rmses_tried})
    return scores_df

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