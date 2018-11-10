import pandas as pd
import math
import numpy as np

def train_and_test_cross_validated(cross_val_lists,
                                   model,
                                   outcome_var,
                                   features):
    all_rmsles = []
    for dictionary in cross_val_lists:
        rmse = train_and_test(dictionary['train_df'],
                               dictionary['dev_df'],
                               model,
                               outcome_var,
                               features)
        all_rmsles.append(rmse)
    return np.mean(all_rmsles)


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

    TODO: rewrite to include cross-validation

    """

    fit_model = model.fit(train_df[features].fillna(0), train_df[outcome_var])
    dev_preds = fit_model.predict(dev_df[features])
    if outcome_var == 'LogSalePrice':
        rmse = rmsle(np.exp(list(dev_df[outcome_var])), np.exp(dev_preds))
    else:
        rmse = rmsle(list(dev_df[outcome_var]), dev_preds)
    return rmse

def try_different_models(cross_val_list,
                         models,
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
    outcome_vars_tried = []
    num_features_tried = []
    features_tried = []
    rmses_tried = []

    for model in models:
        for outcome_var in outcome_vars:
            # TODO: add another layer here
            # of trying different combos
            # of features
            for feature_set in feature_sets:

                rmse = train_and_test_cross_validated(cross_val_list,
                                                      model,
                                                      outcome_var,
                                                      feature_set)

                models_tried.append(model)
                outcome_vars_tried.append(outcome_var)
                features_tried.append(feature_set)
                num_features_tried.append(len(feature_set))
                rmses_tried.append(rmse)

    scores_df = pd.DataFrame(data={'Model': models_tried,
                                   'Outcome Var': outcome_vars_tried,
                                   'Features': features_tried,
                                   'Num Features': num_features_tried,
                                   'Root MSE': rmses_tried})
    return scores_df
