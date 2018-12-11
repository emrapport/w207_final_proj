import pandas as pd
import math
import numpy as np
import logging
import testing_utils as testing

def get_model_dicts_from_data_frame(input_df):
    out_models = []
    for i, row in input_df.iterrows():
        model_dict = {}
        model_dict['model'] = row['Model'][0]
        model_dict['models'] = row['Model']
        model_dict['features'] = row['Features']
        model_dict['outcome_var'] = row['Outcome Var']
        out_models.append(model_dict)
    return out_models

def try_different_voting_ensembles(cross_val_lists,
                                   list_of_model_feature_dicts):
    # note: currently to use this function,
    # all voting ensembles need to have the same predictor (log or not)
    # we should change that....

    ensemble_tried = []
    features_in_each = []
    num_features_in_each = []
    outcome_vars_for_each = []
    rmses = []

    for model_dict_list in list_of_model_feature_dicts:
        models = [mdf['model'] for mdf in model_dict_list]
        feature_lists = [mdf['features'] for mdf in model_dict_list]
        outcome_lists = [mdf['outcome_var'] for mdf in model_dict_list]

        rmse = predict_via_vote_cross_validated(cross_val_lists,
                                                models,
                                                feature_lists,
                                                outcome_lists)

        ensemble_tried.append(models)
        features_in_each.append(feature_lists)
        num_features_in_each.append([len(fl) for fl in feature_lists])
        outcome_vars_for_each.append(outcome_lists)
        rmses.append(rmse)

    scores_df = pd.DataFrame(data={'Models': ensemble_tried,
                                   'Features': features_in_each,
                                   'Num features in each': num_features_in_each,
                                   'Outcome_Vars': outcome_vars_for_each,
                                   'RMSE for ensemble': rmses})
    return scores_df

def predict_via_vote_cross_validated(cross_val_lists,
                                     model_list,
                                     features_list,
                                     outcome_var_list):
    all_test_rmsles = []

    for dictionary in cross_val_lists:
        new_df = predict_via_vote(model_list,
                                  features_list,
                                  outcome_var_list,
                                  dictionary['dev_df'])
        # the ones are a hack to make it not error for train RMSLES
        # since train doesn't matter in this voting world

        #I hard-coded 'SalePrice in here' for third and 5th param
        # since now we're handling exponentiating in predict_via_vote_cross_validated
        # in order to handle ensembles with both outcome types
        test_rmsle = testing.calculate_error([1],
                                             [1],
                                             new_df['SalePrice'],
                                             new_df['vote_by_avg'],
                                             'SalePrice')[0]
        all_test_rmsles.append(test_rmsle)

    #logging.debug("All test RMSLES: {}".format(all_test_rmsles))
    return np.mean(all_test_rmsles)

def predict_via_vote(model_list,
                     feature_set_list,
                     outcome_var_list,
                     dev_df):

    new_dev_df = dev_df.reset_index()

    cols_to_avg = []
    for i, model in enumerate(model_list):
        cols_to_avg.append(i)
        dev_preds = model.predict(new_dev_df[feature_set_list[i]])
        if outcome_var_list[i] == 'LogSalePrice':
            dev_preds = np.exp(dev_preds)
        new_dev_df[i] = dev_preds

    new_dev_df['vote_by_avg'] = (sum([new_dev_df[i] for i in cols_to_avg])
                                / len(cols_to_avg))
    #logging.debug("new dev df vote by avg: {}".format(new_dev_df['vote_by_avg']))
    return new_dev_df
