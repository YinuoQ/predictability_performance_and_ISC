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

def compute_predictability(target_prediction_df):
    predictability_lst = []
    for i in range(3):
        temp_target = target_prediction_df.iloc[i].target
        temp_pred = target_prediction_df.iloc[i].prediction
        if np.sum(temp_target == temp_pred) == 30:
            predictability_lst.append(1)
        elif np.sum(temp_pred == temp_pred[0]) == 30:
            predictability_lst.append(0)
        elif np.sum(temp_target == temp_target[0]) == 30:
            predictability_lst.append(0)
        else:
            predictability_lst.append(np.corrcoef(temp_target, temp_pred)[0,1])
    return np.mean(predictability_lst)

def get_predictability():
    target_prediction_df = pd.DataFrame()

    for role in ['yaw', 'pitch', 'thrust']:
        target_prediction_arr = np.load(f'../../transformer/log/log_{role}_data_seed_1234/lightning_logs/version_0/checkpoints/pred_target.npy')
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
    # import IPython
    # IPython.embed()
    # assert False
    model_formula = "performance ~ predictability"
    model = smf.mixedlm(model_formula, predictability_performance_df, groups=predictability_performance_df['teamID'], re_formula='1+sessionID+communication')
    # Fit the model
    model_result = model.fit()
    print(model_result.summary())

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
   



