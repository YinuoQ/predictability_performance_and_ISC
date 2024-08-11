import os
import cca
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


def get_eeg_performance(lcoation_df, eeg_df):
    eeg_performance_df = copy.deepcopy(eeg_df)
    performance_df = get_performance_from_location(lcoation_df)
    eeg_performance_df['performance'] = None   
    for i in tqdm(range(len(eeg_performance_df))):
        tesm_sess_trial_ring = eeg_performance_df.iloc[i][['teamID', 'sessionID', 'trialID', 'ringID']]
        performance_id = performance_df.loc[(performance_df.teamID == tesm_sess_trial_ring.teamID)
                    & (performance_df.sessionID == tesm_sess_trial_ring.sessionID)
                    & (performance_df.trialID == tesm_sess_trial_ring.trialID)
                    & (performance_df.ringID == tesm_sess_trial_ring.ringID)].index[0]
        eeg_performance_df.at[i, 'performance'] = performance_df.iloc[performance_id].performance
   
    eeg_performance_df.performance = pd.to_numeric(eeg_performance_df.performance)
    return eeg_performance_df


def compute_eeg_ISC(eeg_performance_df):
    eeg_ISC = copy.deepcopy(eeg_performance_df)
    eeg_ISC['eegISC'] = None
    all_eeg = list(eeg_performance_df.yawEEG) + list(eeg_performance_df.pitchEEG) + list(eeg_performance_df.thrustEEG)
    for i in tqdm(range(len(eeg_performance_df))):
        temp_eeg = np.array([eeg_performance_df.iloc[i].yawEEG, eeg_performance_df.iloc[i].pitchEEG, eeg_performance_df.iloc[i].thrustEEG])
        temp_ISC,_ = cca.ISC_eeg(temp_eeg)
        eeg_ISC.at[i, 'eegISC'] = np.abs(temp_ISC[0])
    eeg_ISC = eeg_ISC.dropna().reset_index(drop=True)

    eeg_ISC.eegISC = pd.to_numeric(eeg_ISC.eegISC)
   
    return eeg_ISC

def mixed_effects_model(eeg_ISC_df):
    model_formula = "performance ~ eegISC"
    model = smf.mixedlm(model_formula, eeg_ISC_df, groups=eeg_ISC_df['teamID'])
    # Fit the model
    model_result = model.fit()
    print(model_result.summary())


def get_trialed_EEG_ISC_with_performance(lcoation_df, eeg_df):
    warnings.filterwarnings("ignore")
    performance_df = copy.deepcopy(lcoation_df)
    performance_df = performance_df[['teamID', 'sessionID', 'trialID']].drop_duplicates().reset_index(drop=True)
    performance_df['performance'] = lcoation_df.groupby(['teamID', 'sessionID', 'trialID']).apply(lambda x: x.ringID.max()+1).values
    performance_df['eegISC'] = None
    for i in tqdm(range(len(performance_df))):
        tesm_sess_trial_ring = performance_df.iloc[i][['teamID', 'sessionID', 'trialID']]
        data_ids = eeg_df.loc[(eeg_df.teamID == tesm_sess_trial_ring.teamID)
                    & (eeg_df.sessionID == tesm_sess_trial_ring.sessionID)
                    & (eeg_df.trialID == tesm_sess_trial_ring.trialID)].index
        ISC_epoch_lst = []
        for idx in data_ids:
            temp_eeg = np.array([eeg_df.iloc[i].yawEEG, eeg_df.iloc[i].pitchEEG, eeg_df.iloc[i].thrustEEG])
            temp_ISC,_ = cca.ISC_eeg(temp_eeg)
            ISC_epoch_lst.append(np.abs(temp_ISC[0]))

        performance_df.at[i, 'eegISC'] = np.tanh(np.nanmean(np.arctanh(ISC_epoch_lst)))

    performance_df = performance_df.dropna().reset_index(drop=True)
    performance_df['eegISC'] = pd.to_numeric(performance_df.eegISC)

    return performance_df


if __name__ == '__main__':

    path = '../../data'
    pd.set_option('display.max_columns', None)
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    eeg_df = pd.read_pickle(os.path.join(path, 'epoched_EEG.pkl'))


    # epoch based performances
    eeg_performance_df = get_eeg_performance(lcoation_df, eeg_df)
    eeg_ISC_df = compute_eeg_ISC(eeg_performance_df)
    mixed_effects_model(eeg_ISC_df)

    # trial based performances
    eeg_performance_df = get_trialed_EEG_ISC_with_performance(lcoation_df, eeg_df)
    mixed_effects_model(eeg_performance_df)


