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
    target = np.array(list(target_prediction_df.target)).flatten()
    prediction = np.array(list(target_prediction_df.prediction)).flatten()
    return np.sum(target != prediction) / 90


def get_predictability(seed):
    prediction_target_arr = np.load(f'../../transformer/log/lightning_logs/version_{seed-1}/pred_target.npy')
    target_info = np.load(f'../../transformer/data/seed_{seed}/test/data_info.npy',  allow_pickle=True)
    target_info_df = pd.DataFrame(target_info)
    unique_role_len = int(len(target_info_df)/3)
    target_info_df = target_info_df.rename(columns={0: 'teamID', 1: 'sessionID', 2: 'trialID', 3: 'ringID'})
    predictability_df = copy.deepcopy(target_info_df)[:unique_role_len]
    target_info_df['target'] = list(prediction_target_arr[1].reshape(-1, 30))
    target_info_df['prediction'] = list(prediction_target_arr[0].reshape(-1, 30))
    target_info_df['role'] = ['Yaw'] * unique_role_len + ['Pitch'] * unique_role_len + ['Thrust'] * unique_role_len
    for i in tqdm(range(unique_role_len)):
        temp_tstr = predictability_df.iloc[i]
        three_role_df = target_info_df.loc[(target_info_df.teamID == temp_tstr.teamID) & 
                                           (target_info_df.sessionID == temp_tstr.sessionID)& 
                                           (target_info_df.trialID == temp_tstr.trialID)& 
                                           (target_info_df.ringID == temp_tstr.ringID)]
        temp_team_predictability = compute_predictability(three_role_df)
        if not np.isnan(temp_team_predictability):
            predictability_df.at[i, 'predictability'] = temp_team_predictability
    return predictability_df

def mixed_effects_model(predictability_performance_df):
    model_formula = "performance ~ predictability"
    predictability_performance_df['session'] = predictability_performance_df.sessionID.apply(lambda x: int(x[1:]))
    valid_df = predictability_performance_df
    model = smf.mixedlm(model_formula, valid_df, groups=valid_df['teamID'], re_formula='1 + sessionID')
    model_result = model.fit()
    print(model_result.summary())

def get_predictability_and_performance(performance_df, predictability_df):
    predictability_temp = copy.deepcopy(predictability_df)
    predictability_df = predictability_df[['teamID', 'sessionID', 'trialID', 'ringID']].drop_duplicates().reset_index(drop=True)
    predictability_df['predictability'] = None
    predictability_df['performance'] = None
    predictability_df['communication'] = None
    predictability_df['difficulty'] = None
    for i in tqdm(range(len(predictability_df))):
        temp_pred = predictability_df.iloc[i]
        performance = performance_df.loc[(performance_df.teamID == temp_pred.teamID) & 
                           (performance_df.sessionID == temp_pred.sessionID)& 
                           (performance_df.trialID == temp_pred.trialID)& 
                           (performance_df.ringID == temp_pred.ringID)].iloc[0]
        predictability = predictability_temp.loc[(performance_df.teamID == temp_pred.teamID) & 
                           (performance_df.sessionID == temp_pred.sessionID)& 
                           (performance_df.trialID == temp_pred.trialID)& 
                           (performance_df.ringID == temp_pred.ringID)].predictability.mean()
        predictability_df.at[i, 'predictability'] = predictability
        predictability_df.at[i, 'performance'] = performance.performance
        predictability_df.at[i, 'communication'] = performance.communication
        predictability_df.at[i, 'difficulty'] = performance.difficulty
    predictability_df.performance = pd.to_numeric(predictability_df.performance)
    predictability_df.predictability = pd.to_numeric(predictability_df.predictability)

    return predictability_df.dropna().reset_index(drop=True)

def get_trial_performance(lcoation_df, predictability_df):
    performance_df = copy.deepcopy(lcoation_df)
    performance_df = performance_df[['teamID', 'sessionID', 'trialID']].drop_duplicates().reset_index(drop=True)
    performance_df['performance'] = lcoation_df.groupby(['teamID', 'sessionID', 'trialID']).apply(lambda x: x.ringID.max()+1).values
    performance_df['predictability'] = None
    for i in tqdm(range(len(performance_df))):
        tesm_sess_trial = performance_df.iloc[i][['teamID', 'sessionID', 'trialID']]
        data_ids = predictability_df.loc[(predictability_df.teamID == tesm_sess_trial.teamID)
                    & (predictability_df.sessionID == tesm_sess_trial.sessionID)
                    & (predictability_df.trialID == tesm_sess_trial.trialID)].index
        if len(data_ids) > 0:
            predictability_epoch_lst = []
            for idx in data_ids:
                predictability_epoch_lst.append(predictability_df.iloc[idx].predictability)
            performance_df.at[i, 'predictability'] = np.mean(predictability_epoch_lst)
  
    a = performance_df.dropna().reset_index(drop=True)
    a.predictability = pd.to_numeric(a.predictability)
    model_formula = "performance ~ predictability"
    model = smf.mixedlm(model_formula, a, groups=a['teamID'], re_formula='1+sessionID')
    model_result = model.fit()
    print(model_result.summary())


if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    warnings.filterwarnings('ignore')
    # epoch based performances
    performance_df = get_performance(lcoation_df)
    predictability_df = pd.DataFrame()
    for seed in [1,2,3]:
        predictability_df = pd.concat((predictability_df, get_predictability(seed)))
    predictability_df = predictability_df.reset_index(drop=True)
    pred_perf_df = get_predictability_and_performance(performance_df, predictability_df)
    # import IPython
    # IPython.embed()
    # assert False
    mixed_effects_model(pred_perf_df)

    # trial based performances
    trial_based_performance = get_trial_performance(lcoation_df, predictability_df)










   