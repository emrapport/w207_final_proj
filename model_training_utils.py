import pandas as pd
import math
import numpy as np
import logging
import testing_utils as testing

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

    test_rmse, train_rmse = testing.calculate_error(train_df[outcome_var],
                                                    train_preds,
                                                    dev_df[outcome_var],
                                                    dev_preds,
                                                    outcome_var)

    return test_rmse, train_rmse, fit_model
