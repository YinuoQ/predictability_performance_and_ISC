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


def get_pupil_performance(lcoation_df, pupil_df):
    pupil_performance_df = copy.deepcopy(pupil_df)
    performance_df = get_performance_from_location(lcoation_df)
    pupil_performance_df['performance'] = None 

    for i in tqdm(range(len(pupil_performance_df))):
        tesm_sess_trial_ring = pupil_performance_df.iloc[i][['teamID', 'sessionID', 'trialID', 'ringID']]
        try:
            performance_id = performance_df.loc[(performance_df.teamID == tesm_sess_trial_ring.teamID)
                        & (performance_df.sessionID == tesm_sess_trial_ring.sessionID)
                        & (performance_df.trialID == tesm_sess_trial_ring.trialID)
                        & (performance_df.ringID == tesm_sess_trial_ring.ringID)].index[0]
        except:
            continue           
        pupil_performance_df.at[i, 'performance'] = performance_df.iloc[performance_id].performance
   
    pupil_performance_df.performance = pd.to_numeric(pupil_performance_df.performance)
    return pupil_performance_df

def ISC_among_pupil(pupil_df):
    cov_mat = np.abs(np.corrcoef(np.array([pupil_df.yawPupil, 
                                           pupil_df.pitchPupil, 
                                           pupil_df.thrustPupil])))
    r_xy = cov_mat[0,1]
    r_xz = cov_mat[0,2]
    r_yz = cov_mat[1,2]

    z_xy = np.arctanh(r_xy)
    z_xz = np.arctanh(r_xz)
    z_yz = np.arctanh(r_yz)
    team_corr_coeff = np.tanh(np.nanmean([z_xy, z_xz, z_yz]))
    return team_corr_coeff

def compute_pupil_ISC(pupil_performance_df):
    pupil_ISC = copy.deepcopy(pupil_performance_df)
    pupil_ISC['pupilISC'] = None

    for i in range(len(pupil_ISC)):
        temp_ISC = ISC_among_pupil(pupil_performance_df.iloc[i])
        pupil_ISC.at[i, 'pupilISC'] = temp_ISC
    pupil_ISC = pupil_ISC.dropna().reset_index(drop=True)

    pupil_ISC.pupilISC = pd.to_numeric(pupil_ISC.pupilISC)
   
    return pupil_ISC

def get_trialed_pupil_ISC_with_performance(lcoation_df, pupil_df):
    warnings.filterwarnings("ignore")
    performance_df = copy.deepcopy(lcoation_df)
    performance_df = performance_df[['teamID', 'sessionID', 'trialID']].drop_duplicates().reset_index(drop=True)
    performance_df['performance'] = lcoation_df.groupby(['teamID', 'sessionID', 'trialID']).apply(lambda x: x.ringID.max()+1).values
    performance_df['pupilISC'] = None
    for i in tqdm(range(len(performance_df))):
        tesm_sess_trial_ring = performance_df.iloc[i][['teamID', 'sessionID', 'trialID']]
        data_ids = pupil_df.loc[(pupil_df.teamID == tesm_sess_trial_ring.teamID)
                    & (pupil_df.sessionID == tesm_sess_trial_ring.sessionID)
                    & (pupil_df.trialID == tesm_sess_trial_ring.trialID)].index
        ISC_epoch_lst = []
        for idx in data_ids:
            ISC_epoch_lst.append(ISC_among_pupil(pupil_df.iloc[idx]))

        performance_df.at[i, 'pupilISC'] = np.tanh(np.nanmean(np.arctanh(ISC_epoch_lst)))

    performance_df = performance_df.dropna().reset_index(drop=True)
    performance_df['pupilISC'] = pd.to_numeric(performance_df.pupilISC)

    return performance_df

def mixed_effects_model(pupil_ISC_df):
    model_formula = "performance ~ pupilISC"
    model = smf.mixedlm(model_formula, pupil_ISC_df, groups=pupil_ISC_df['teamID'])
    # Fit the model
    model_result = model.fit()
    print(model_result.summary())

if __name__ == '__main__':
    path = '../../data'
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    pupil_df = pd.read_pickle(os.path.join(path, 'epoched_pupil.pkl'))
    
    pd.set_option('display.max_columns', None)

    # epoch based performances
    pupil_performance_df = get_pupil_performance(lcoation_df, pupil_df)
    pupil_ISC_df = compute_pupil_ISC(pupil_performance_df)
    mixed_effects_model(pupil_ISC_df)

    # trial based performances
    pupil_performance_df = get_trialed_pupil_ISC_with_performance(lcoation_df, pupil_df)
    mixed_effects_model(pupil_performance_df)



