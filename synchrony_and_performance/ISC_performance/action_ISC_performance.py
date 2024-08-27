import os
import sys
import copy
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from scipy.stats import linregress
import statsmodels.formula.api as smf
sys.path.insert(1, '../preprocessing/')
from performance_from_location import get_performance_from_location


def get_action_performance(lcoation_df, action_df):
    action_performance_df = copy.deepcopy(action_df)
    performance_df = get_performance_from_location(lcoation_df)
    action_performance_df['performance'] = None   
    for i in tqdm(range(len(action_performance_df))):
        tesm_sess_trial_ring = action_performance_df.iloc[i][['teamID', 'sessionID', 'trialID', 'ringID']]
        performance_id = performance_df.loc[(performance_df.teamID == tesm_sess_trial_ring.teamID)
                    & (performance_df.sessionID == tesm_sess_trial_ring.sessionID)
                    & (performance_df.trialID == tesm_sess_trial_ring.trialID)
                    & (performance_df.ringID == tesm_sess_trial_ring.ringID)].index[0]
        action_performance_df.at[i, 'performance'] = performance_df.iloc[performance_id].performance
   
    action_performance_df.performance = pd.to_numeric(action_performance_df.performance)
    return action_performance_df

def get_trialed_action_ISC_with_performance(lcoation_df, action_df):
    warnings.filterwarnings("ignore") 
    performance_df = copy.deepcopy(lcoation_df)
    performance_df = performance_df[['teamID', 'sessionID', 'trialID']].drop_duplicates().reset_index(drop=True)
    performance_df['performance'] = lcoation_df.groupby(['teamID', 'sessionID', 'trialID']).apply(lambda x: x.ringID.max()+1).values
    performance_df['actionISC'] = None
    for i in tqdm(range(len(performance_df))):
        tesm_sess_trial = performance_df.iloc[i][['teamID', 'sessionID', 'trialID']]
        data_ids = action_df.loc[(action_df.teamID == tesm_sess_trial.teamID)
                    & (action_df.sessionID == tesm_sess_trial.sessionID)
                    & (action_df.trialID == tesm_sess_trial.trialID)].index
        ISC_epoch_lst = []
        for idx in data_ids:
            ISC_epoch_lst.append(ISC_among_actions(action_df.iloc[idx]))

        performance_df.at[i, 'actionISC'] = np.nanmean(ISC_epoch_lst)

    performance_df = performance_df.dropna().reset_index(drop=True)
    performance_df['actionISC'] = pd.to_numeric(performance_df.actionISC)

    return performance_df

def ISC_among_actions(action_data, epoch_based=True):
    action_arr = np.array([action_data.yawAction, 
                           action_data.pitchAction, 
                           action_data.thrustAction])
    
    cov_mat = np.abs(np.corrcoef(action_arr))
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
        temp_ISC = ISC_among_actions(action_performance_df.iloc[i], True)
        action_ISC.at[i, 'actionISC'] = temp_ISC
    action_ISC = action_ISC.dropna().reset_index(drop=True)

    action_ISC.actionISC = pd.to_numeric(action_ISC.actionISC)
   
    return action_ISC

def mixed_effects_model(action_ISC_df):
    model_formula = "performance ~ actionISC"
    print(model_formula)
    model = smf.mixedlm(model_formula, action_ISC_df, groups=action_ISC_df['teamID'])
    # Fit the model
    model_result = model.fit()
    print(model_result.summary())

if __name__ == '__main__':

    path = '../../data'
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    action_df = pd.read_pickle(os.path.join(path, 'epoched_action.pkl'))
    
    pd.set_option('display.max_columns', None)
    # epoched based performance
    action_performance_df = get_action_performance(lcoation_df, action_df)
    action_ISC_df = compute_action_ISC(action_performance_df)
    mixed_effects_model(action_ISC_df)
    
    # trial based performances
    action_performance_df = get_trialed_action_ISC_with_performance(lcoation_df, action_df)
    # import IPython
    # IPython.embed()
    # assert False
    # fig, ax = plt.subplots(3,6, figsize=(30, 10),sharey=True)
    # plt.tight_layout(pad=2)
    # teamID_lst = action_performance_df.teamID.unique()
    # for i in range(3):
    #     for j in range(6):
    #         y = action_performance_df[action_performance_df.teamID == teamID_lst[i*6+j]].performance
    #         x = action_performance_df[action_performance_df.teamID == teamID_lst[i*6+j]].actionISC
    #         linregress_result = linregress(x,y)
    #         ax[i,j].plot(x,y, 'o')
    #         ax[i,j].plot([0,1], np.array([0,1])*linregress_result.slope+linregress_result.intercept)
    #         ax[i,j].text(0.5,1,f"r-value = {format(linregress_result.rvalue, '.3f')}")
    #         ax[i,j].text(0.5,2,f"slope = {format(linregress_result.slope, '.3f')}")
    #         ax[i,j].set_title(f"{teamID_lst[i*6+j]}")
    #         ax[i,0].set_ylabel("performance")
    # plt.savefig('action_performance.png')
    mixed_effects_model(action_performance_df)




