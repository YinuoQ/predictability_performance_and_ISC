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
    
    input_data_lst.append(np.array(list(data_df.iloc[start_id:end_id][f'{role}Action']))[:,:])

    return input_data_lst 

def reformat_input_output_data(input_output_lst):
    eeg_pp_action_speech_loc = []
    input_output_arr = np.array(input_output_lst, dtype=object)
    

    for i in [0,2,4,6]:
        # Reshape the array to have 10 rows (each row corresponds to one of the 9 groups)
        reshaped_arr_9th = input_output_arr[i::10]
        reshaped_arr_10th = input_output_arr[i+1::10]

        # Stack the 9th and 10th elements row-wise
        if i == 0:
            stacked_arr = np.concatenate((np.vstack(reshaped_arr_9th)[:, None, :,:], np.vstack(reshaped_arr_10th)[:, None, :,:]), axis=1)
        else:
            stacked_arr = np.concatenate((np.vstack(reshaped_arr_9th)[:, None,:], np.vstack(reshaped_arr_10th)[:, None,:]), axis=1)
        # Concatenate the stacked arrays along the first axis
        eeg_pp_action_speech_loc.append(stacked_arr)
    eeg_pp_action_speech_loc.append(np.vstack(input_output_arr[8::10]))
    
    return eeg_pp_action_speech_loc, np.vstack(input_output_arr[9::10])

def generate_training_testing_val_dataset(data_df, seed=1, data_split_ratio=(0.5, 0.45, 0.05)):
    # import IPython
    # IPython.embed()
    # assert False    
    unique_team = data_df[['teamID', 'sessionID']].drop_duplicates().reset_index(drop=True)
    role_lst = ['yaw', 'pitch', 'thrust']
    all_role_train_input = []
    all_role_test_input = []
    all_role_val_input = []    
    all_role_train_output = []
    all_role_test_output = []
    all_role_val_output = []
    all_team_sess_ring = []
    for role in role_lst:
        training_lst = []
        testing_lst = []
        validation_lst = []
        test_team_sess_trial_ring_lst = []
        for team_id in unique_team.iterrows():
            temp_data_df = data_df[(data_df.teamID == team_id[1].teamID)&(data_df.sessionID == team_id[1].sessionID)]
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
        all_role_train_input.append(training_arr_input)
        all_role_test_input.append(testing_arr_input)
        all_role_val_input.append(validation_arr_input)   
        all_role_train_output.append(training_arr_output)
        all_role_test_output.append(testing_arr_output)
        all_role_val_output.append(validation_arr_output)
        all_team_sess_ring.append(test_team_sess_trial_ring_df)

    mkdir(os.path.join('train'))
    mkdir(os.path.join('test'))
    mkdir(os.path.join('validation'))
    for i, modality in enumerate(['EEG', 'Pupil', 'Action', 'Speech', 'location']):
        np.save(os.path.join('train', f'train_{modality.lower()}.npy'), np.concatenate([all_role_train_input[0][i], all_role_train_input[1][i], all_role_train_input[2][i]]))
        np.save(os.path.join('test', f'test_{modality.lower()}.npy'), np.concatenate([all_role_test_input[0][i], all_role_test_input[1][i], all_role_test_input[2][i]]))
        np.save(os.path.join('validation', f'validation_{modality.lower()}.npy'), np.concatenate([all_role_val_input[0][i], all_role_val_input[1][i], all_role_val_input[2][i]]))

    np.save(os.path.join('train', 'train_output.npy'), np.concatenate([all_role_train_output[0], all_role_train_output[1], all_role_train_output[2]]))
    np.save(os.path.join('test', 'test_output.npy'), np.concatenate([all_role_test_output[0], all_role_test_output[1], all_role_test_output[2]]))
    np.save(os.path.join('validation', 'validation_output.npy'), np.concatenate([all_role_val_output[0], all_role_val_output[1], all_role_val_output[2]]))
    np.save(os.path.join('test', 'data_info.npy'), pd.concat(all_team_sess_ring).reset_index(drop=True))

if __name__ == '__main__':
    path = '../../data'
    seed = 1234

    pd.set_option('display.max_columns', None)
    action_df, location_df, pupil_df, eeg_df, speech_df, ring_df, performance_df = read_data(path)
    transformer_data_df = transformer_data_generator(action_df, location_df, pupil_df, eeg_df, speech_df, ring_df, performance_df)   
    generate_training_testing_val_dataset(transformer_data_df, seed)
    








