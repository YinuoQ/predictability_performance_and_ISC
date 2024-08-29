import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt


def read_data(path):
    action_df = pd.read_pickle(os.path.join(path, 'epoched_action.pkl'))
    location_df = pd.read_pickle(os.path.join(path, 'epoched_location.pkl'))
    pupil_df = pd.read_pickle(os.path.join(path, 'epoched_pupil.pkl'))
    eeg_df = pd.read_pickle(os.path.join(path, 'epoched_eeg.pkl'))
    speech_df = pd.read_pickle(os.path.join(path, 'epoched_speech_event.pkl'))
    
    return action_df, location_df, pupil_df, eeg_df, speech_df

def transformer_data_generator(action_df, location_df, pupil_df, eeg_df, speech_df):
    eeg_df['yawAction'] = None
    eeg_df['pitchAction'] = None
    eeg_df['thrustAction'] = None
    eeg_df['yawSpeech'] = None
    eeg_df['pitchSpeech'] = None
    eeg_df['thrustSpeech'] = None
    eeg_df['yawPupil'] = None
    eeg_df['pitchPupil'] = None
    eeg_df['thrustPupil'] = None
    eeg_df['location'] = None
    for i in tqdm(range(len(eeg_df))):
        team_sess_trial_ring = eeg_df.iloc[i][['teamID', 'sessionID', 'trialID', 'ringID']]
        query_string = f"teamID == '{team_sess_trial_ring[0]}' and sessionID == '{team_sess_trial_ring[1]}' \
                        and trialID == {team_sess_trial_ring[2]} and ringID == {team_sess_trial_ring[3]}"
        temp_action = action_df.query(query_string)
        temp_speech = speech_df.query(query_string)
        temp_pupil = pupil_df.query(query_string)
        temp_location = location_df.query(query_string)
        eeg_df.at[i, 'location'] = temp_location.location.iloc[0]
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
    for modality in ['EEG', 'Pupil', 'Action', 'Speech', 'location']:
        if modality == 'EEG':
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{input_role[0]}{modality}']))[:,:,:256])
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{input_role[1]}{modality}']))[:,:,:256])
        elif modality == 'location':
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{modality}'])))

        else:
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{input_role[0]}{modality}']))[:,:60])
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{input_role[1]}{modality}']))[:,:60])
    
    input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{role}Action']))[:,60:])

    return input_data_lst 

def reformat_input_output_data(input_output_lst):
    eeg_pp_action_speech = []
    input_output_arr = np.array(input_output_lst, dtype=object)

    for i in [0,2,4,6]:
        # Reshape the array to have 10 rows (each row corresponds to one of the 9 groups)
        reshaped_arr_9th = input_output_arr[i::10]
        reshaped_arr_10th = input_output_arr[i+1::10]

        # Stack the 9th and 10th elements row-wise
        stacked_arr = np.vstack((reshaped_arr_9th, reshaped_arr_10th)).T

        # Flatten the stacked array back to 1D
        reformed_arr = stacked_arr.flatten()

        # Combine pairs of arrays and stack them
        stacked_arrays = []
        for j in range(0, len(reformed_arr), 2):
            combined = np.stack((reformed_arr[j], reformed_arr[j+1]), axis=1)
            stacked_arrays.append(combined)

        # Concatenate the stacked arrays along the first axis
        output_array = np.concatenate(stacked_arrays, axis=0)
        eeg_pp_action_speech.append(output_array)
    eeg_pp_action_speech.append(np.vstack(input_output_arr[8::10]))
    
    return eeg_pp_action_speech, np.vstack(input_output_arr[9::10])

def shuffle_arr(input_arr, output_arr, seed, test_team_sess=None):
   
    shuffled_input_lst = []
    np.random.seed(seed=seed)
    random_idx = np.random.choice(len(input_arr[0]), len(input_arr[0]), replace=False)
    for i in range(len(input_arr)):
        shuffled_input_lst.append(input_arr[i][random_idx])
    shuffled_output = output_arr#[random_idx]
    if test_team_sess is not None:
        return shuffled_input_lst, shuffled_output, test_team_sess.iloc[random_idx]
    else:
        return shuffled_input_lst, shuffled_output

def generate_training_testing_val_dataset(data_df, seed=1, data_split_ratio=(0.75, 0.2, 0.05)):
    unique_team = data_df['teamID'].unique()

    training_lst = []
    testing_lst = []
    validation_lst = []
    test_team_sess_trial_ring_lst = []
    role_lst = ['yaw', 'pitch', 'thrust']

    for role in role_lst:
        for team_id in unique_team:
            temp_data_df = data_df[data_df.teamID == team_id]
            shuffled_df = temp_data_df.sample(frac=1, random_state=seed)
            train_end = int(len(shuffled_df)*data_split_ratio[0])
            test_end = train_end+int(len(shuffled_df)*data_split_ratio[1])
            val_len = len(shuffled_df)
            temp_training_lst = select_model_input_and_output(shuffled_df, 0, train_end, role)
            temp_testing_lst = select_model_input_and_output(shuffled_df, train_end, test_end, role)
            temp_validation_lst = select_model_input_and_output(shuffled_df, test_end, val_len, role)
            training_lst += temp_training_lst
            testing_lst += temp_testing_lst
            validation_lst += temp_validation_lst
            test_team_sess_trial_ring_lst.append(temp_data_df[['teamID', 'sessionID', 'trialID', 'ringID']].iloc[train_end:test_end])
  
    test_team_sess_trial_ring_df = pd.concat(test_team_sess_trial_ring_lst)

    training_arr_input, training_arr_output = reformat_input_output_data(training_lst)
    testing_arr_input, testing_arr_output = reformat_input_output_data(testing_lst)
    validation_arr_input, validation_arr_output = reformat_input_output_data(validation_lst)

    training_arr_input, training_arr_output = shuffle_arr(training_arr_input, training_arr_output, seed)
    testing_arr_input, testing_arr_output, test_team_sess_trial_ring_df = shuffle_arr(testing_arr_input, testing_arr_output, seed, test_team_sess_trial_ring_df)
    validation_arr_input, validation_arr_output = shuffle_arr(validation_arr_input, validation_arr_output, seed)
  
    for i, modality in enumerate(['EEG', 'Pupil', 'Action', 'Speech', 'location']):

        np.save(os.path.join('train', f'train_{modality.lower()}.npy'), training_arr_input[i])
        np.save(os.path.join('test', f'test_{modality.lower()}.npy'), testing_arr_input[i])
        np.save(os.path.join('validation', f'validation_{modality.lower()}.npy'), validation_arr_input[i])

    # np.save(os.path.join('debug_data', 'train', f'train_output.npy'), training_arr_output)
    # np.save(os.path.join('debug_data', 'test', f'test_output.npy'), testing_arr_output)
    # np.save(os.path.join('debug_data', 'validation', f'validation_output.npy'), validation_arr_output)
    # np.save(os.path.join('debug_data', 'test', f'data_info.npy'), np.array(test_team_sess_trial_ring_lst[:len(unique_team)], dtype=object))

if __name__ == '__main__':
    path = '../../data'
    seed = 1234

    pd.set_option('display.max_columns', None)
    action_df, location_df, pupil_df, eeg_df, speech_df = read_data(path)
    transformer_data_df = transformer_data_generator(action_df, location_df, pupil_df, eeg_df, speech_df)   
    generate_training_testing_val_dataset(transformer_data_df, seed)
    









