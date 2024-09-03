import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.insert(1, '../utils')
from common import mkdir

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
    eeg_df['performance'] = None
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

        eeg_df.at[i, 'location'] = np.concatenate((temp_location.location.iloc[0], np.array(temp_ring[['ringX', 'ringY', 'ringZ']]).T), axis=1)
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

def select_model_input_and_output(data_df, start_id, end_id): 
    input_data_lst = [] 
    for modality in ['EEG', 'Pupil', 'Action', 'Speech', 'location']:
        if modality == 'EEG':
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'yaw{modality}']))[:,:,:256])
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'pitch{modality}']))[:,:,:256])
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'thrust{modality}']))[:,:,:256])
        elif modality == 'location':
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{modality}'])))
        else:
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'yaw{modality}']))[:,:60])
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'pitch{modality}']))[:,:60])
            input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'thrust{modality}']))[:,:60])

    return input_data_lst, np.array(list(data_df.iloc[start_id:end_id]['performance']))

def reformat_input_output_data(input_output_lst):
    eeg_pp_action_speech_loc = []
    input_output_arr = np.array(input_output_lst, dtype=object)

    for i in [0,3,6,9]:
        # Reshape the array to have 10 rows (each row corresponds to one of the 9 groups)
        reshaped_arr_9th = input_output_arr[i::13]
        reshaped_arr_10th = input_output_arr[i+1::13]
        reshaped_arr_11th = input_output_arr[i+2::13]

        # Stack the 9th and 10th elements row-wise
        if i == 0:
            stacked_arr = np.concatenate((np.vstack(reshaped_arr_9th)[:, None, :,:], np.vstack(reshaped_arr_10th)[:, None, :,:], np.vstack(reshaped_arr_11th)[:, None, :,:]), axis=1)
        else:
            stacked_arr = np.concatenate((np.vstack(reshaped_arr_9th)[:, None,:], np.vstack(reshaped_arr_10th)[:, None,:], np.vstack(reshaped_arr_11th)[:, None,:]), axis=1)
        # Concatenate the stacked arrays along the first axis
        eeg_pp_action_speech_loc.append(stacked_arr)
    eeg_pp_action_speech_loc.append(np.vstack(input_output_arr[12::13]))
    
    return eeg_pp_action_speech_loc

def generate_training_testing_val_dataset(data_df, seed=1, data_split_ratio=(0.75, 0.2, 0.05)):
    unique_team = data_df['teamID'].unique()

    role_lst = ['yaw', 'pitch', 'thrust']

    # for role in role_lst:
    training_lst = []
    testing_lst = []
    validation_lst = []
    training_output_lst = []
    testing_output_lst = []
    validation_output_lst = []
    test_team_sess_trial_ring_lst = []
    for team_id in unique_team:
        temp_data_df = data_df[data_df.teamID == team_id]
        shuffled_df = temp_data_df.sample(frac=1, random_state=seed)
        train_end = int(len(shuffled_df)*data_split_ratio[0])
        test_end = train_end+int(len(shuffled_df)*data_split_ratio[1])
        val_len = len(shuffled_df)
        temp_training_lst, training_performance = select_model_input_and_output(shuffled_df, 0, train_end)
        temp_testing_lst, testing_performance = select_model_input_and_output(shuffled_df, train_end, test_end)
        temp_validation_lst, validation_performance = select_model_input_and_output(shuffled_df, test_end, val_len)
        training_lst += temp_training_lst
        testing_lst += temp_testing_lst
        validation_lst += temp_validation_lst
        training_output_lst.append(training_performance)
        testing_output_lst.append(testing_performance)
        validation_output_lst.append(validation_performance)
        test_team_sess_trial_ring_lst.append(temp_data_df[['teamID', 'sessionID', 'trialID', 'ringID']].iloc[train_end:test_end])

    test_team_sess_trial_ring_df = pd.concat(test_team_sess_trial_ring_lst)

    training_arr_input = reformat_input_output_data(training_lst)
    testing_arr_input = reformat_input_output_data(testing_lst)
    validation_arr_input = reformat_input_output_data(validation_lst)
    import IPython
    IPython.embed()
    assert False
    training_arr_output = np.concatenate(training_output_lst)
    testing_arr_output = np.concatenate(testing_output_lst)
    validation_arr_output = np.concatenate(validation_output_lst)

    mkdir(os.path.join('train'))
    mkdir(os.path.join('test'))
    mkdir(os.path.join('validation'))
    for i, modality in enumerate(['EEG', 'Pupil', 'Action', 'Speech', 'location']):
        np.save(os.path.join('train', f'train_{modality.lower()}.npy'), training_arr_input[i])
        np.save(os.path.join('test', f'test_{modality.lower()}.npy'), testing_arr_input[i])
        np.save(os.path.join('validation', f'validation_{modality.lower()}.npy'), validation_arr_input[i])

    np.save(os.path.join('train', f'train_output.npy'), training_arr_output)
    np.save(os.path.join('test', f'test_output.npy'), testing_arr_output)
    np.save(os.path.join('validation', f'validation_output.npy'), validation_arr_output)
    np.save(os.path.join('test', f'data_info.npy'), np.array(test_team_sess_trial_ring_lst[:len(unique_team)], dtype=object))

if __name__ == '__main__':
    path = '../../data'
    seed = 1234

    pd.set_option('display.max_columns', None)
    action_df, location_df, pupil_df, eeg_df, speech_df, ring_df, performance_df = read_data(path)
    transformer_data_df = transformer_data_generator(action_df, location_df, pupil_df, eeg_df, speech_df, ring_df, performance_df)   
    generate_training_testing_val_dataset(transformer_data_df, seed)
    









