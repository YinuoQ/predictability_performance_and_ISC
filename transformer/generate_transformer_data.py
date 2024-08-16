import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_data(path):
    action_df = pd.read_pickle(os.path.join(path, 'epoched_action.pkl'))
    pupil_df = pd.read_pickle(os.path.join(path, 'epoched_pupil.pkl'))
    eeg_df = pd.read_pickle(os.path.join(path, 'epoched_eeg.pkl'))
    speech_df = pd.read_pickle(os.path.join(path, 'epoched_speech_event.pkl'))
    return action_df, pupil_df, eeg_df, speech_df

def transformer_data_generator(action_df, pupil_df, eeg_df, speech_df):
    eeg_df['yawAction'] = None
    eeg_df['pitchAction'] = None
    eeg_df['thrustAction'] = None
    eeg_df['yawSpeech'] = None
    eeg_df['pitchSpeech'] = None
    eeg_df['thrustSpeech'] = None
    eeg_df['yawPupil'] = None
    eeg_df['pitchPupil'] = None
    eeg_df['thrustPupil'] = None
    for i in tqdm(range(len(eeg_df))):
        team_sess_trial_ring = eeg_df.iloc[i][['teamID', 'sessionID', 'trialID', 'ringID']]
        query_string = f"teamID == '{team_sess_trial_ring[0]}' and sessionID == '{team_sess_trial_ring[1]}' \
                        and trialID == {team_sess_trial_ring[2]} and ringID == {team_sess_trial_ring[3]}"
        temp_action = action_df.query(query_string)
        temp_speech = speech_df.query(query_string)
        temp_pupil = pupil_df.query(query_string)
        if len(temp_action) == 1:
            eeg_df.at[i, 'yawAction'] = temp_action.yawAction.iloc[0]
            eeg_df.at[i, 'pitchAction'] = temp_action.pitchAction.iloc[0]
            eeg_df.at[i, 'thrustAction'] = temp_action.thrustAction.iloc[0]
        if len(temp_speech) == 1:
            eeg_df.at[i, 'yawSpeech'] = temp_speech.yawSpeech.iloc[0]
            eeg_df.at[i, 'pitchSpeech'] = temp_speech.pitchSpeech.iloc[0]
            eeg_df.at[i, 'thrustSpeech'] = temp_speech.thrustSpeech.iloc[0]
        if len(temp_pupil) == 1:
            eeg_df.at[i, 'yawPupil'] = temp_pupil.yawPupil.iloc[0]
            eeg_df.at[i, 'pitchPupil'] = temp_pupil.pitchPupil.iloc[0]
            eeg_df.at[i, 'thrustPupil'] = temp_pupil.thrustPupil.iloc[0]
    return eeg_df.dropna().reset_index(drop=True)

def shuffle_data(arr, seed):
    np.random.seed(seed)
    np.random.shuffle(arr)
    return arr

def train_test_val_split(input_arr, team_id_arr, data_split_ratio):
    num_rows = input_arr.shape[0]
    train_end_idx = int(num_rows * data_split_ratio[0])
    val_end_idx = int(num_rows * data_split_ratio[1]) + train_end_idx
    train_arr = input_arr[:train_end_idx, :]
    val_arr = input_arr[train_end_idx:val_end_idx, :]
    test_arr = input_arr[val_end_idx:, :]
    test_team_id_arr = team_id_arr[val_end_idx:, :]
    return train_arr, val_arr, test_arr, test_team_id_arr

def select_model_input_and_output(data_df, start_id, end_id, role):

    role_2_id = {'yaw': 0,
                'pitch': 1,
                'thrust': 2}

    id_2_role = {0: 'yaw',
                 1: 'pitch',
                 2: 'thrust'}    
    all_role_val = list(range(3))
    all_role_val.remove(role_2_id[role])
    input_role = [id_2_role[x] for x in all_role_val]
    input_data_lst = []
    for modality in ['EEG', 'Pupil', 'Action', 'Speech']:
        if modality == 'EEG':
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{input_role[0]}{modality}']))[:,:,:256])
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{input_role[1]}{modality}']))[:,:,:256])
        else:
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{input_role[0]}{modality}']))[:,:60])
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{input_role[1]}{modality}']))[:,:60])
    
    input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{role}Action']))[:,60:])
    input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{role}Action']))[:,60:])

    return input_data_lst 

def generate_training_testing_val_dataset(data_df, seed=1, data_split_ratio=(0.75, 0.2, 0.05)):
    unique_team_sess_trial = data_df['teamID'].unique()

    training_lst = []
    testing_lst = []
    validation_lst = []
    team_sess_trial_lst = []
    role_lst = ['yaw', 'pitch', 'thrust']
    for role in role_lst:
        for team_id in unique_team_sess_trial:
            temp_data_df = data_df[data_df.teamID == team_id]
            shuffled_df = temp_data_df.sample(frac=1, random_state=seed)
            train_end = int(len(shuffled_df)*data_split_ratio[0])
            test_end = train_end+int(len(shuffled_df)*data_split_ratio[1])
            val_len = len(shuffled_df)
            temp_training_lst = select_model_input_and_output(shuffled_df, 0, train_end, role)
            temp_testing_lst = select_model_input_and_output(shuffled_df, train_end, test_end, role)
            temp_validation_lst = select_model_input_and_output(shuffled_df, test_end, val_len, role)
            training_lst.append(temp_training_lst)
            testing_lst.append(temp_testing_lst)
            validation_lst.append(temp_validation_lst)
            
    import IPython
    IPython.embed()
    assert False
if __name__ == '__main__':
    path = '../data'
    seed = 1234

    pd.set_option('display.max_columns', None)
    action_df, pupil_df, eeg_df, speech_df = read_data(path)
    transformer_data_df = transformer_data_generator(action_df, pupil_df, eeg_df, speech_df)   
    generate_training_testing_val_dataset(transformer_data_df, seed)
    
    transformer_data_df.to_pickle(os.path.join(path, 'transformer_data.pkl'))









