import os
import sys
import copy
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
sys.path.insert(1, '../preprocessing/')
from performance_from_location import get_performance_from_location

def get_performance(lcoation_df):
    performance_df = get_performance_from_location(lcoation_df)
    return performance_df

def normalized_cross_correlation(a, b):
    # Ensure the two series are numpy arrays
    a = np.array(a)
    b = np.array(b)
    
    # Mean of the series
    mean_a = np.mean(a)
    mean_b = np.mean(b)
    
    # Subtract the mean from each series
    a_diff = a - mean_a
    b_diff = b - mean_b
    
    # Calculate the numerator as the sum of the element-wise product of the two series
    numerator = np.sum(a_diff * b_diff)
    
    # Calculate the denominator as the product of the square roots of the sum of squares
    denominator = np.sqrt(np.sum(a_diff ** 2)) * np.sqrt(np.sum(b_diff ** 2))
    
    # Return the normalized cross-correlation
    return numerator / denominator if denominator != 0 else 0

def compute_predictability(target_prediction_df):
    predictability_lst = []
    for i in range(3):
        temp_target = target_prediction_df.iloc[i].target
        temp_pred = target_prediction_df.iloc[i].prediction
        # predictability_lst.append(np.corrcoef(temp_target, temp_pred)[0,1])
        # predictability_lst.append(np.sum((temp_pred - temp_target)**2)/len(temp_pred))
        predictability_lst.append(normalized_cross_correlation(temp_target, temp_pred))
    return np.tanh(np.nanmean(np.arctanh(predictability_lst)))

def get_predictability():
    target_prediction_df = pd.DataFrame()

    for i, role in enumerate(['yaw', 'pitch', 'thrust']):
        target_prediction_arr = np.load(f'../../transformer/log_data_seed_1234/lightning_logs/version_{i+3}/pred_target.npy')
        target_info = np.load(f'../../transformer/data/test/{role}/data_info.npy',  allow_pickle=True)
        target_info_df = pd.concat(target_info)
        target_info_df['role'] = role
        target_info_df['target'] = list(target_prediction_arr[1].reshape(-1, 30))
        target_info_df['prediction'] = list(target_prediction_arr[0].reshape(-1, 30))
        target_prediction_df = pd.concat((target_prediction_df, target_info_df), ignore_index=True).reset_index(drop=True)
    
    predictabiltiy_df = copy.deepcopy(target_prediction_df[['teamID', 'sessionID', 'trialID', 'ringID']].drop_duplicates().reset_index(drop=True))
    predictabiltiy_df['predictability'] = None

    for i in range(len(predictabiltiy_df)):
        temp_tstr = predictabiltiy_df.iloc[i]
        three_role_df = target_prediction_df.loc[(target_prediction_df.teamID == temp_tstr.teamID) & 
                                                 (target_prediction_df.sessionID == temp_tstr.sessionID)& 
                                                 (target_prediction_df.trialID == temp_tstr.trialID)& 
                                                 (target_prediction_df.ringID == temp_tstr.ringID)]
        temp_team_predictability = compute_predictability(three_role_df)
        predictabiltiy_df.at[i, 'predictability'] = temp_team_predictability

    return predictabiltiy_df
def mixed_effects_model(predictability_performance_df):

    model_formula = "performance ~ predictability"
    valid_df = predictability_performance_df.dropna().reset_index(drop=True)
    model = smf.mixedlm(model_formula, valid_df, groups=valid_df['teamID'], re_formula='sessionID')
    # Fit the model
    model_result = model.fit()
    print(model_result.summary())
    import IPython
    IPython.embed()
    assert False
def get_predictability_and_performance(performance_df, predictability_df):

    predictability_df['performance'] = None
    predictability_df['communication'] = None
    predictability_df['difficulty'] = None
    for i in tqdm(range(len(predictability_df))):
        temp_pred = predictability_df.iloc[i]
        performance = performance_df.loc[(performance_df.teamID == temp_pred.teamID) & 
                           (performance_df.sessionID == temp_pred.sessionID)& 
                           (performance_df.trialID == temp_pred.trialID)& 
                           (performance_df.ringID == temp_pred.ringID)].iloc[0]
        predictability_df.at[i, 'performance'] = performance.performance
        predictability_df.at[i, 'communication'] = performance.communication
        predictability_df.at[i, 'difficulty'] = performance.difficulty
    predictability_df.performance = pd.to_numeric(predictability_df.performance)
    predictability_df.predictability = pd.to_numeric(predictability_df.predictability)
    return predictability_df



if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))

    # epoch based performances
    performance_df = get_performance(lcoation_df)
    predictability_df = get_predictability()
    pred_perf_df = get_predictability_and_performance(performance_df, predictability_df)
    mixed_effects_model(pred_perf_df)
   