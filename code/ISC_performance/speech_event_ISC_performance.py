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


def get_speech_performance(lcoation_df, speech_event):
    performance_df = get_performance_from_location(lcoation_df)
    speech_performance_df = copy.deepcopy(speech_event)
    speech_performance_df['performance'] = None 
    speech_performance_df = speech_performance_df[speech_performance_df.communication != 'No'].reset_index(drop=True)
    for i in tqdm(range(len(speech_performance_df))):
        tesm_sess_trial_ring = speech_performance_df.iloc[i][['teamID', 'sessionID', 'trialID', 'ringID']]
        try:
            performance_id = performance_df.loc[(performance_df.teamID == tesm_sess_trial_ring.teamID)
                        & (performance_df.sessionID == tesm_sess_trial_ring.sessionID)
                        & (performance_df.trialID == tesm_sess_trial_ring.trialID)
                        & (performance_df.ringID == tesm_sess_trial_ring.ringID)].index[0]
        except:
            continue           
        speech_performance_df.at[i, 'performance'] = performance_df.iloc[performance_id].performance
   
    speech_performance_df.performance = pd.to_numeric(speech_performance_df.performance)
    return speech_performance_df

def ISC_among_speech(speech_event):
    cov_mat = np.abs(np.corrcoef(np.array([speech_event.yawSpeech, 
                                           speech_event.pitchSpeech, 
                                           speech_event.thrustSpeech])))
    r_xy = cov_mat[0,1]
    r_xz = cov_mat[0,2]
    r_yz = cov_mat[1,2]

    z_xy = np.arctanh(r_xy)
    z_xz = np.arctanh(r_xz)
    z_yz = np.arctanh(r_yz)
    team_corr_coeff = np.tanh(np.nanmean([z_xy, z_xz, z_yz]))
    return team_corr_coeff

def compute_speech_ISC(speech_performance_df):
    speech_ISC = copy.deepcopy(speech_performance_df)
    speech_ISC['speechISC'] = None

    for i in range(len(speech_ISC)):
        temp_ISC = ISC_among_speech(speech_performance_df.iloc[i])
        speech_ISC.at[i, 'speechISC'] = temp_ISC
    speech_ISC = speech_ISC.dropna().reset_index(drop=True)

    speech_ISC.speechISC = pd.to_numeric(speech_ISC.speechISC)
   
    return speech_ISC

def mixed_effects_model(speech_ISC_df):
    model_formula = "performance ~ speechISC"
    model = smf.mixedlm(model_formula, speech_ISC_df, groups=speech_ISC_df['teamID'])
    # Fit the model
    model_result = model.fit()
    print(model_result.summary())

def get_trialed_speech_ISC_with_performance(lcoation_df, speech_df):
    warnings.filterwarnings("ignore")
    performance_df = copy.deepcopy(lcoation_df)
    performance_df = performance_df[['teamID', 'sessionID', 'trialID']].drop_duplicates().reset_index(drop=True)
    performance_df['performance'] = lcoation_df.groupby(['teamID', 'sessionID', 'trialID']).apply(lambda x: x.ringID.max()+1).values
    performance_df['speechISC'] = None
    for i in tqdm(range(len(performance_df))):
        tesm_sess_trial_ring = performance_df.iloc[i][['teamID', 'sessionID', 'trialID']]
        data_ids = speech_df.loc[(speech_df.teamID == tesm_sess_trial_ring.teamID)
                    & (speech_df.sessionID == tesm_sess_trial_ring.sessionID)
                    & (speech_df.trialID == tesm_sess_trial_ring.trialID)].index
        ISC_epoch_lst = []
        for idx in data_ids:
            ISC_epoch_lst.append(ISC_among_speech(speech_df.iloc[idx]))

        performance_df.at[i, 'speechISC'] = np.tanh(np.nanmean(np.arctanh(ISC_epoch_lst)))

    performance_df = performance_df.dropna().reset_index(drop=True)
    performance_df['speechISC'] = pd.to_numeric(performance_df.speechISC)

    return performance_df

if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    speech_event = pd.read_pickle(os.path.join(path, 'epoched_speech_event.pkl'))
   
    # epoch based performances
    speech_performance_df = get_speech_performance(lcoation_df, speech_event)
    speech_ISC_df = compute_speech_ISC(speech_performance_df)
    mixed_effects_model(speech_ISC_df)
   
    # trial based performances
    speech_performance_df = get_trialed_speech_ISC_with_performance(lcoation_df, speech_event)
    mixed_effects_model(speech_performance_df)




