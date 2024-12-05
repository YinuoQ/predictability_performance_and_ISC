import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from statsmodels.stats.anova import AnovaRM 
from action_performance import get_predictability, get_performance
sys.path.insert(1, '../ISC_performance/')
from pupil_ISC_performance import get_trialed_pupil_ISC_with_performance
from eeg_ISC_performance import get_trialed_EEG_ISC_with_performance
from action_ISC_performance import get_trialed_action_ISC_with_performance
from performance_from_location import get_performance_from_location

def get_ISC_and_reatings(path):
    helpfulness_df = pd.read_csv(os.path.join(path, 'helpfulness.csv'))
    familiarity_df = pd.read_csv(os.path.join(path, 'familiarity.csv'))
    location_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    pupil_df = pd.read_pickle(os.path.join(path, 'epoched_pupil.pkl'))
    eeg_df = pd.read_pickle(os.path.join(path, 'epoched_eeg.pkl'))
    action_df = pd.read_pickle(os.path.join(path, 'epoched_action.pkl'))
    action_df = get_trialed_action_ISC_with_performance(location_df, action_df)    
    eeg_df = get_trialed_EEG_ISC_with_performance(location_df, eeg_df)
    pupil_df = get_trialed_pupil_ISC_with_performance(location_df, pupil_df)
    local_performance_df = get_performance_from_location(location_df)
    local_performance_df = local_performance_df.drop(columns=['communication',
       'difficulty', 'ringX', 'ringY', 'ringZ', 'location', 'time', 'startTime', 'ringTime']) 
    global_performance_df = location_df[['teamID', 'sessionID', 'trialID']].drop_duplicates().reset_index(drop=True)
    global_performance_df['performance'] = location_df.groupby(['teamID', 'sessionID', 'trialID']).apply(lambda x: x.ringID.max()+1).values
    
    return helpfulness_df, familiarity_df, pupil_df, eeg_df, action_df, local_performance_df, global_performance_df


def append_one_df_2_another(df_main, df2, key_name):
    df_main[key_name] = None
    for i in tqdm(range(len(df_main))):
        team_str, sess_str = df_main.iloc[i][['teamID', 'sessionID']]
        temp_fam = df2.loc[(df2.team == int(team_str[1:]))
                         & (df2.session == int(sess_str[1:]))].iloc[0][key_name]
        df_main.at[i, key_name] = temp_fam
    
    return df_main

def familiarity_2_performance(familiarity_df, local_performance_df, global_performance_df):
    local_performance_df = append_one_df_2_another(local_performance_df, familiarity_df, 'familiarity')
    global_performance_df = append_one_df_2_another(global_performance_df, familiarity_df, 'familiarity')

    print('local performance <- familiarity')
    local_performance_df.performance = pd.to_numeric(local_performance_df['performance'])
    local_performance_df.familiarity = pd.to_numeric(local_performance_df['familiarity'])
    model_formula = "performance~familiarity"
    model = smf.mixedlm(model_formula, local_performance_df, groups=local_performance_df['teamID'], re_formula='sessionID+1')
    model_result = model.fit()
    print(model_result.summary())


    print('global performance <- familiarity')
    global_performance_df.performance = pd.to_numeric(global_performance_df['performance'])
    global_performance_df.familiarity = pd.to_numeric(global_performance_df['familiarity'])
    model_formula = "performance~familiarity"
    model = smf.mixedlm(model_formula, global_performance_df, groups=global_performance_df['teamID'], re_formula='sessionID+1')
    model_result = model.fit()
    print(model_result.summary())


def helpfulness_2_performance(helpfulness_df, local_performance_df, global_performance_df):
    local_performance_df = append_one_df_2_another(local_performance_df, helpfulness_df, 'helpfulness')
    global_performance_df = append_one_df_2_another(global_performance_df, helpfulness_df, 'helpfulness')

    print('local performance <- helpfulness')
    local_performance_df.performance = pd.to_numeric(local_performance_df['performance'])
    local_performance_df.helpfulness = pd.to_numeric(local_performance_df['helpfulness'])
    model_formula = "performance~helpfulness"
    model = smf.mixedlm(model_formula, local_performance_df, groups=local_performance_df['teamID'], re_formula='sessionID+1')
    model_result = model.fit()
    print(model_result.summary())


    print('global performance <- helpfulness')
    global_performance_df.performance = pd.to_numeric(global_performance_df['performance'])
    global_performance_df.helpfulness = pd.to_numeric(global_performance_df['helpfulness'])
    model_formula = "performance~helpfulness"
    model = smf.mixedlm(model_formula, global_performance_df, groups=global_performance_df['teamID'], re_formula='sessionID+1')
    model_result = model.fit()
    print(model_result.summary())


def predictability_2_helpfulness(helpfulness_df, predictability_df):
    predictability_helpfulness_df = append_one_df_2_another(predictability_df, helpfulness_df, 'helpfulness')
    print('predictability <- helpfulness')
    predictability_helpfulness_df.predictability = pd.to_numeric(predictability_helpfulness_df['predictability'])
    predictability_helpfulness_df.helpfulness = pd.to_numeric(predictability_helpfulness_df['helpfulness'])
    model_formula = "predictability~helpfulness"
    model = smf.mixedlm(model_formula, predictability_helpfulness_df, groups=predictability_helpfulness_df['teamID'], re_formula='sessionID+1')
    model_result = model.fit()
    print(model_result.summary())

def familiarity_2_eeg(familiarity_df, eeg_df):
    eeg_familiarity_df = append_one_df_2_another(eeg_df, familiarity_df, 'familiarity')
    print('eeg <- familiarity')
    eeg_familiarity_df.eegISC = pd.to_numeric(eeg_familiarity_df['eegISC'])
    eeg_familiarity_df.familiarity = pd.to_numeric(eeg_familiarity_df['familiarity'])
    model_formula = "eegISC~familiarity"
    model = smf.mixedlm(model_formula, eeg_familiarity_df, groups=eeg_familiarity_df['teamID'], re_formula='sessionID+1')
    model_result = model.fit()
    print(model_result.summary())



def helpfulness_2_action(helpfulness_df, action_df):
    action_helpfulness_df = append_one_df_2_another(action_df, helpfulness_df, 'helpfulness')
    print('action <- helpfulness')
    action_helpfulness_df.actionISC = pd.to_numeric(action_helpfulness_df['actionISC'])
    action_helpfulness_df.helpfulness = pd.to_numeric(action_helpfulness_df['helpfulness'])
    model_formula = "actionISC~helpfulness"
    model = smf.mixedlm(model_formula, action_helpfulness_df, groups=action_helpfulness_df['teamID'], re_formula='sessionID+1')
    model_result = model.fit()
    print(model_result.summary())

def familiarity_2_predictability(familiarity_df, predictability_df):
    predictability_familiarity_df = append_one_df_2_another(predictability_df, familiarity_df, 'familiarity')
    print('predictability <- familiarity')
    predictability_familiarity_df.predictability = pd.to_numeric(predictability_familiarity_df['predictability'])
    predictability_familiarity_df.familiarity = pd.to_numeric(predictability_familiarity_df['familiarity'])
    model_formula = "predictability~familiarity"
    model = smf.mixedlm(model_formula, predictability_familiarity_df, groups=predictability_familiarity_df['teamID'], re_formula='sessionID+1')
    model_result = model.fit()
    print(model_result.summary())

    import IPython
    IPython.embed()
    assert False




if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    predictability_df = pd.DataFrame()
    for seed in [1,2,3, 4]:
        temp_pred= get_predictability(seed)
        predictability_df = pd.concat((predictability_df, temp_pred))
    predictability_df = predictability_df.reset_index(drop=True)
    helpfulness_df, familiarity_df, pupil_df, eeg_df, action_df, local_performance_df, global_performance_df= get_ISC_and_reatings(path)

    familiarity_2_performance(familiarity_df, local_performance_df, global_performance_df)
    helpfulness_2_performance(helpfulness_df, local_performance_df, global_performance_df)
    helpfulness_2_action(helpfulness_df, action_df)
    predictability_2_helpfulness(helpfulness_df, predictability_df)
    familiarity_2_eeg(familiarity_df, eeg_df)
    familiarity_2_predictability(familiarity_df, predictability_df)





