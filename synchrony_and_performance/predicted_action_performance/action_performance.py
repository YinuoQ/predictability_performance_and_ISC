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
def compute_predictability(target_prediction_arr, target_info):
    target_info_df = pd.concat(target_info, ignore_index=True)
    reshaped_targ_pred_arr = np.reshape(target_prediction_arr, (2, 3,-1,30))
    if len(target_info_df) == reshaped_targ_pred_arr.shape[2]:
        target_info_df['predictability'] = None
        for i in range(len(target_info_df)):
            temp_arr = reshaped_targ_pred_arr[:,:,i,:] # targ, role (y, p, t), time
            team_predictability = np.sum(temp_arr[0] == temp_arr[1]) / temp_arr[0].size
            target_info_df.at[i, 'predictability'] = team_predictability
            # # plot close
            # fig, ax = plt.subplots(1,3, figsize=(10, 3))
            # for j in range(3):
            #     ax[j].plot(range(30), temp_arr[0][j], color=f'C0')
            #     ax[j].plot(range(30), temp_arr[1][j], '--', color=f'C1', alpha=0.8)
            # plt.savefig(f"plots/{i}.png")
            # plt.close()
    else:
        print('size miss match, check input data and try again')
        assert False
    return target_info_df

def mixed_effects_model(predictability_performance_df):
    model_formula = "performance ~ predictability"
    model = smf.mixedlm(model_formula, predictability_performance_df, groups=predictability_performance_df['teamID'])
    # Fit the model
    model_result = model.fit()
    print(model_result.summary())

def get_predictability_and_performance(performance_df, predictability_df):
    import IPython
    IPython.embed()
    assert False
    predictability_df['performance'] = None
    for i in range(len(predictability_df)):
        temp_pred = predictability_df.iloc[i]
        performance = performance_df.loc[(performance_df.teamID == temp_pred.teamID) & 
                           (performance_df.sessionID == temp_pred.sessionID)& 
                           (performance_df.trialID == temp_pred.trialID)& 
                           (performance_df.ringID == temp_pred.ringID)].performance.iloc[0]
        predictability_df.at[i, 'performance'] = performance
    predictability_df.performance = pd.to_numeric(predictability_df.performance)
    predictability_df.predictability = pd.to_numeric(predictability_df.predictability)




if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    target_prediction_arr = np.load('../../transformer/log_data_seed_1/lightning_logs/version_0/pred_target.npy')
    target_info =  np.load('../../transformer/data/test/data_info.npy',  allow_pickle=True)

    # epoch based performances
    performance_df = get_performance(lcoation_df)
    predictability_df = compute_predictability(target_prediction_arr, target_info)
    pred_perf_df = get_predictability_and_performance(performance_df, predictability_df)

    # mixed_effects_model(speech_ISC_df)
   
    # # trial based performances
    # speech_performance_df = get_trialed_speech_ISC_with_performance(lcoation_df, speech_event)
    # mixed_effects_model(speech_performance_df)




