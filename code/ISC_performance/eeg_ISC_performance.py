import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
sys.path.insert(1, '../preprocessing/')
from performance_from_location import get_performance_from_location


def get_action_performance(path):
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    eeg_df = pd.read_pickle(os.path.join(path, 'epoched_EEG.pkl'))
    import IPython
    IPython.embed()
    assert False

    performance_df = get_performance_from_location(lcoation_df)
    eeg_df['performance'] = None   
    for i in tqdm(range(len(eeg_df))):
        tesm_sess_trial_ring = eeg_df.iloc[i][['teamID', 'sessionID', 'trialID', 'ringID']]
        performance_id = performance_df.loc[(performance_df.teamID == tesm_sess_trial_ring.teamID)
                    & (performance_df.sessionID == tesm_sess_trial_ring.sessionID)
                    & (performance_df.trialID == tesm_sess_trial_ring.trialID)
                    & (performance_df.ringID == tesm_sess_trial_ring.ringID)].index[0]
        eeg_df.at[i, 'performance'] = performance_df.iloc[performance_id].performance
   
    eeg_df.performance = pd.to_numeric(eeg_df.performance)
    return eeg_df

def ISC_among_actions(action_data):
    cov_mat = np.abs(np.corrcoef(np.array([action_data.yawAction, 
                                           action_data.pitchAction, 
                                           action_data.thrustAction])))
    r_xy = cov_mat[0,1]
    r_xz = cov_mat[0,2]
    r_yz = cov_mat[1,2]

    z_xy = np.arctanh(r_xy)
    z_xz = np.arctanh(r_xz)
    z_yz = np.arctanh(r_yz)
    team_corr_coeff = np.tanh(np.nanmean([z_xy, z_xz, z_yz]))
    return team_corr_coeff

def compute_action_ISC(action_performance_df):
    action_ISC = copy.deepcopy(action_performance_df)
    action_ISC['actionISC'] = None

    for i in range(len(action_performance_df)):
        temp_ISC = ISC_among_actions(action_performance_df.iloc[i])
        action_ISC.at[i, 'actionISC'] = temp_ISC
    action_ISC = action_ISC.dropna().reset_index(drop=True)

    action_ISC.actionISC = pd.to_numeric(action_ISC.actionISC)
   
    return action_ISC

def mixed_effects_model(action_ISC_df):
    action_ISC_performance = copy.deepcopy(action_ISC_df)
    action_ISC_performance = action_ISC_performance.drop(columns={'yawAction', 'pitchAction','thrustAction'})
    model_formula = "performance ~ actionISC"
    model = smf.mixedlm(model_formula, action_ISC_performance, groups=action_ISC_performance['teamID'])
    # Fit the model
    model_result = model.fit()
    print(model_result.summary())

if __name__ == '__main__':

    path = '../../data'
    pd.set_option('display.max_columns', None)
    action_performance_df = get_action_performance(path)
    action_ISC_df = compute_action_ISC(action_performance_df)
    mixed_effects_model(action_ISC_df)




