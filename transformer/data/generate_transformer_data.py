import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.insert(1, '../utils')
from common import mkdir
from sklearn.model_selection import train_test_split

def read_data(path):
    action_df = pd.read_pickle(os.path.join(path, 'epoched_action.pkl'))
    location_df = pd.read_pickle(os.path.join(path, 'epoched_location.pkl'))
    pupil_df = pd.read_pickle(os.path.join(path, 'epoched_pupil.pkl'))
    eeg_df = pd.read_pickle(os.path.join(path, 'epoched_eeg.pkl'))
    speech_df = pd.read_pickle(os.path.join(path, 'epoched_speech_event.pkl'))
    ring_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    performance_df = pd.read_pickle(os.path.join(path, 'performance.pkl'))
    return action_df, location_df, pupil_df, eeg_df, speech_df, ring_df, performance_df

def transformer_data_generator(action_df, location_df, pupil_df, eeg_df, speech_df, ring_df, performance_df):
    eeg_df = eeg_df.assign(yawAction=None, pitchAction=None, thrustAction=None, yawSpeech=None, 
                           pitchSpeech=None, thrustSpeech=None, yawPupil=None, pitchPupil=None, 
                           thrustPupil=None, location=None, performance=None)
    for i in tqdm(range(len(eeg_df))):
        team_sess_trial_ring = eeg_df.iloc[i][['teamID', 'sessionID', 'trialID', 'ringID']]
        query_string = f"teamID == '{team_sess_trial_ring[0]}' and sessionID == '{team_sess_trial_ring[1]}' \
                        and trialID == {team_sess_trial_ring[2]} and ringID == {team_sess_trial_ring[3]}"
        temp_action = action_df.query(query_string)
        temp_speech = speech_df.query(query_string)
        temp_pupil = pupil_df.query(query_string)
        temp_location = location_df.query(query_string)
        temp_ring = ring_df.query(query_string)
        temp_performance = performance_df.query(query_string)

        if not temp_location.empty and not temp_ring.empty:
            eeg_df.at[i, 'location'] = np.concatenate((temp_location.location.iloc[0], np.array(temp_ring[['ringX', 'ringY', 'ringZ']]).T), axis=1)
        if not temp_performance.empty:    
            eeg_df.at[i, 'performance'] = temp_performance.performance.iloc[0]

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


def select_model_input_and_output(data_df, role, is_test=False):
    # Map role names to indices
    role_lst = ['yaw', 'pitch', 'thrust']

    # Define modalities and input columns excluding the specified role
    modalities = {
        'EEG': {'columns': [f'{r}EEG' for r in role_lst if r != role], 'slice': (0, 256)},
        'Pupil': {'columns': [f'{r}Pupil' for r in role_lst if r != role], 'slice': (0, 60)},
        'Action': {'columns': [f'{r}Action' for r in role_lst if r != role], 'slice': (0, 60)},
        'Speech': {'columns': [f'{r}Speech' for r in role_lst if r != role], 'slice': (0, 60)},
        'location': {'columns': ['location'], 'slice': None}  # No slicing needed for location
    }
 
    # Collect input data based on modality, excluding the specified role
    input_data_lst = []
    for modality, details in modalities.items():
        two_subj_lst = []
        for col in details['columns']:
            data_slice = np.array(list(data_df[col]))
            if details['slice']:  # Apply slicing if specified
                if modality == 'EEG':
                    data_slice = data_slice[:, :, details['slice'][0]:details['slice'][1]]
                else:
                    data_slice = data_slice[:, details['slice'][0]:details['slice'][1]]
            two_subj_lst.append(data_slice)
        if len(two_subj_lst) > 1 and modality == 'EEG':
            input_data_lst.append(np.transpose(two_subj_lst, (1, 0, 2, 3)))
        elif len(two_subj_lst) > 1:
            input_data_lst.append(np.transpose(two_subj_lst, (1, 0, 2)))

        else:
            input_data_lst.append(two_subj_lst[0])
        

    # Collect output data based on the specified role
    output_data = np.array(list(data_df[f'{role}Action']))[:, 30:]

    if not is_test:
        return input_data_lst, output_data
    else:
        return input_data_lst, output_data, data_df[['teamID', 'sessionID', 'trialID', 'ringID', 'communication','difficulty']]
def split_data(data_df, seed, data_split_ratio=(0.7, 0.25, 0.05), n_splits=4):   
    role_lst = ['yaw', 'pitch', 'thrust']
    shuffled_df = copy.deepcopy(data_df)
    np.random.seed(seed)
    shuffled_df = shuffled_df.sample(frac=1).reset_index(drop=True)  # Shuffle sessions

    # Divide unique sessions into 4 distinct, non-overlapping test groups
    group_size = len(data_df) // n_splits
    test_groups = [shuffled_df.iloc[i*group_size: (i+1)*group_size] for i in range(n_splits)]

    for role in role_lst:
        train_lst, test_lst, val_lst, test_team_sess_trial_ring_lst = [], [], [], []
        for split_round in range(n_splits):
            # Define the test, validation, and training groups for this split
            test_sessions = test_groups[split_round]
            # Define the remaining data as the training + validation pool (75% of the data)

            remaining_sessions = pd.concat([test_groups[i] for i in range(n_splits) if i != split_round])


            # Split the remaining 75% into training (70%) and validation (5%)
            train_sessions, val_sessions = train_test_split(
                remaining_sessions, 
                test_size=data_split_ratio[2] / (data_split_ratio[0] + data_split_ratio[2]), 
                random_state=seed + split_round
            )
            
            # Collect data for each set
            train_in, train_out = select_model_input_and_output(train_sessions, role)
            save_data(train_in, train_out, role, split_round, 'train')

            test_in, test_out, test_sessions_info = select_model_input_and_output(test_sessions, role, is_test=True)
            save_data(test_in, test_out, role, split_round, 'test', test_sessions_info)


            val_in, val_out = select_model_input_and_output(val_sessions, role)
            save_data(val_in, val_out, role, split_round, 'validation')

            # Save the datasets

def save_data(input_lst, output_lst, role, split_round, ttv, test_sessions_info=None):
    base_path = os.path.join(role, f'split_{split_round}')
    mkdir(os.path.join(base_path, ttv))

    for i, modality in enumerate(['EEG', 'Pupil', 'Action', 'Speech', 'location']):
        np.save(os.path.join(base_path, ttv, f'{ttv}_{modality.lower()}.npy'), input_lst[i])

    np.save(os.path.join(base_path, ttv, f'{ttv}_output.npy'), output_lst)
    if ttv == 'test':
        np.save(os.path.join(base_path, 'test', 'test_session_info.npy'), test_sessions_info)


if __name__ == '__main__':
    path = '../../data'
    seed = 1
    pd.set_option('display.max_columns', None)
    action_df, location_df, pupil_df, eeg_df, speech_df, ring_df, performance_df = read_data(path)
    transformer_data_df = transformer_data_generator(action_df, location_df, pupil_df, eeg_df, speech_df, ring_df, performance_df)   

    split_data(transformer_data_df, seed)
    








