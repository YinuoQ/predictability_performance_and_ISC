import sys
import os
import re
import copy
import glob
import pyxdf
import random
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, filtfilt, butter, lfilter

def read_txt(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines
def read_xdf(path):
    data, header = pyxdf.load_xdf(path)
    return data

def find_string_in_list(strings_list, target_string):
    for string in strings_list:
        if string in target_string:
            return string
    return None

def get_data_modality_and_role(channel_name, channel_type, desktop_name):
    role_lst = ['Yaw', 'Pitch', 'Thrust']
    role = find_string_in_list(role_lst, channel_name)
    if role != None:
        # the channel name contains one of the roles
        modality = channel_name.replace(role, "")
        desktop_role_key = [desktop_name, role]
    else:
        # it is speech, use client ID instead of roles
        modality = channel_type
        role = channel_name
        desktop_role_key = None

    return modality, role, desktop_role_key



def filter_rows_by_turbulence_condition(data_frame, column_name, threshold):
    """
    Filter rows from the given data frame based on a condition for rows with modality equal to 'turbulence'.
    Parameters:
        data_frame (pandas.DataFrame): The input data frame.
        column_name (str): The name of the column containing 2D lists.
        threshold (float): The threshold for the sum of absolute values of elements in the 2D lists.
    Returns:
        pandas.DataFrame: A new data frame containing rows that meet the filtering condition.
    """
    # Create a list to store the rows that meet the condition
    filtered_rows = []
    # Iterate through each row of the data frame
    for index, row in data_frame.iterrows():
        # Get the value of the "modality" column for the current row
        modality_value = row["modality"]
        # Check if the modality is 'turbulence'
        if modality_value == 'turbulence':
            # Calculate the sum of absolute values of elements in the 2D list
            two_d_list = row[column_name]
            list_sum_abs = sum(abs(item) for sublist in two_d_list for item in sublist)
            # Check if the sum of absolute values is less than the threshold
            if list_sum_abs < threshold:
                # If the condition is met, skip this row (remove the row with modality='turbulence' and abs(sum(data)) < threshold)
                continue
        # If the condition is not met, add the entire row to the filtered list
        filtered_rows.append(row)
    # Create a new data frame containing only the filtered rows
    filtered_df = pd.DataFrame(filtered_rows)
    return filtered_df

def map_roles_based_on_desktop(data_frame, desktop_role_key_lst):
    """
    Map roles based on the desktop column using a predefined list.
    
    Parameters:
        data_frame (pandas.DataFrame): The input data frame.
        desktop_role_key_lst (list): A 2D list containing mappings of desktop names to roles.
        
    Returns:
        pandas.DataFrame: A new data frame with updated roles based on the mappings.
    """
    # Create a copy of the original data frame to store the updated roles
    updated_df = data_frame.copy()
    
    # List of roles to pass (do not change)
    roles_to_pass = ["Yaw", "Pitch", "Thrust"]
    
    # Iterate through each row of the data frame
    for index, row in updated_df.iterrows():
        role = row["role"]
        desktop = row["desktop"]
        
        # Check if the role is in roles_to_pass
        if role in roles_to_pass:
            # If the role is in roles_to_pass, skip this row
            continue
        
        # Find the corresponding role in desktop_role_key_lst based on the desktop value
        for desktop_mapping in desktop_role_key_lst:
            if desktop_mapping[0] == desktop:
                updated_role = desktop_mapping[1]
                break
        else:
            # If the desktop value is not found in desktop_role_key_lst, keep the original role
            updated_role = role
        
        # Update the role in the new data frame
        updated_df.at[index, "role"] = updated_role
    
    return updated_df


def generate_data_frame(team_lst, session_lst, modality_lst, role_lst, desktop_name_lst, data_lst, data_time_lst, desktop_role_key_lst):
    # first, convert lists to a dataframe
    output_df = pd.DataFrame(
    {'teamID': team_lst, 
     'sessionID': session_lst, 
     'modality': modality_lst,
     'role': role_lst,
     'desktop': desktop_name_lst,
     'data':data_lst,
     'time':data_time_lst,
    })

    output_df = filter_rows_by_turbulence_condition(output_df, 'data', 1)
    output_df = map_roles_based_on_desktop(output_df, desktop_role_key_lst)
    # remove turbulence recorded on server (redundent)
    output_df = output_df[output_df["role"].str.contains("Turbulence") == False]
    return output_df

def remove_cache_data(data, team_session_names):
    modality_lst = []
    role_lst = []
    desktop_name_lst = []
    data_lst = []
    data_time_lst = []
    desktop_role_key_lst = []
    team_lst = []
    session_lst = []
    for i in range(len(data)):
        try:
            data[i]['time_series'].flatten()
        except:
            continue
        if np.sum(data[i]['time_series'] == -1) < len(data[i]['time_series'].flatten()):
            # is not cache
            channel_name = data[i]['info']['name'][0]
            channel_type = data[i]['info']['type'][0]
            desktop_name = data[i]['info']['hostname'][0]
            if channel_name not in ['YawLocation', 'PitchLocation', 'ThrustLocation', 'BAlert']:
                continue
            modality, role, desktop_role_key = get_data_modality_and_role(channel_name, channel_type, desktop_name)
            modality_lst.append(modality)
            role_lst.append(role)
            desktop_name_lst.append(desktop_name)
            data_lst.append(data[i]['time_series'])
            data_time_lst.append(data[i]['time_stamps'])
            team_lst.append(team_session_names[0])
            session_lst.append(team_session_names[1])
            if desktop_role_key != None and desktop_role_key not in desktop_role_key_lst:
                desktop_role_key_lst.append(desktop_role_key)

    data_df = generate_data_frame(team_lst, session_lst, modality_lst, 
                                  role_lst, desktop_name_lst, 
                                  data_lst, data_time_lst, desktop_role_key_lst)

    return data_df

def merge_xdf(xdf_name_lst, team_session_names):
    """
    Some experiment sessions contians two .xdf files.
    This function will merge the two files and output one dataframe
    """
    temp_xdf0 = read_xdf(xdf_name_lst[0])
    temp_xdf1 = read_xdf(xdf_name_lst[1])
    
    data_0 = remove_cache_data(temp_xdf0, team_session_names)
    data_1 = remove_cache_data(temp_xdf1, team_session_names)
    
    if len(xdf_name_lst) == 3:
        temp_xdf2 = read_xdf(xdf_name_lst[2])
        data_2 = remove_cache_data(temp_xdf2, team_session_names)


    for i in range(len(data_0)):
        data_0_time = data_0.iloc[i]['time']
        data_0_data = data_0.iloc[i]['data']
        data_1_df = data_1.loc[(data_1['modality'] == data_0.iloc[i]['modality'])
                  &(data_1['role'] == data_0.iloc[i]['role'])]

        if len(data_1_df) == 0:
            continue
        data_1_time = data_1_df.iloc[0]['time']
        data_1_data = data_1_df.iloc[0]['data']
        
        if data_1_time[0] >= data_0_time[-1]:
            data_time = np.append(data_0_time, data_1_time)
            data_data = np.append(data_0_data, data_1_data, axis=0)
        else:
            data_time = np.append(data_1_time, data_0_time)
            data_data = np.append(data_1_data, data_0_data, axis=0)

        if len(xdf_name_lst) == 3:
            data_2_df = data_2.loc[(data_2['modality'] == data_0.iloc[i]['modality'])
                      &(data_2['role'] == data_0.iloc[i]['role'])]

            data_2_time = data_2_df.iloc[0]['time']
            data_2_data = data_2_df.iloc[0]['data']
        

            if data_2_time[0] >= data_1_time[-1]:
                data_time = np.append(data_time, data_2_time)
                data_data = np.append(data_data, data_2_data, axis=0)
            else:
                data_time = np.append(data_2_time, data_time)
                data_data = np.append(data_2_data, data_data, axis=0)

        data_0.iat[i, -1] = data_time
        data_0.iat[i, -2] = data_data
    
    return data_0

def merge_bronen_dataframes(data_df):
    for team in data_df['teamID'].unique():
        for session in data_df['sessionID'].unique():
            for modality in data_df['modality'].unique():
                for role in data_df['role'].unique():
                    team_session_data = data_df.loc[(data_df['teamID'] == team)
                                               &(data_df['sessionID'] == session)
                                               &(data_df['modality'] == modality)
                                               &(data_df['role'] == role)]
                    if len(team_session_data) > 1:
                        merged_dic = {}
                        # check time and see which go first
                        chunck_1 = team_session_data.iloc[0]
                        chunck_2 = team_session_data.iloc[1]
                        chunck_1_start_time = chunck_1['time'][0]
                        chunck_2_start_time = chunck_2['time'][0]
                        merged_dic['teamID'] = team
                        merged_dic['sessionID'] = session
                        merged_dic['modality'] = modality
                        merged_dic['role'] = role
                        merged_dic['desktop'] = chunck_1['desktop']
                        if chunck_2_start_time > chunck_1_start_time:
                            # chunk 2 after chunk 1
                            for key in ['data', 'time']:
                                merged_dic[key] = [np.append(chunck_1[key], chunck_2[key])]
                        else:
                            # chunk 2 before chunk 1
                            for key in ['data', 'time']:
                                merged_dic[key] = [np.append(chunck_2[key], chunck_1[key])]
                        
                        merged_df = pd.DataFrame.from_dict(merged_dic)
                        data_df.loc[(data_df['teamID'] == team)
                                   &(data_df['sessionID'] == session)
                                   &(data_df['modality'] == modality)
                                   &(data_df['role'] == role)].iloc[0] = merged_df
                        data_df = data_df.drop(team_session_data.index[1])

def sort_data_frame(data_df):
    sorted_data_df = data_df.sort_values(by=['teamID', 'sessionID', 'role']).reset_index(drop=True)
    return sorted_data_df

def process_file(input_path, output_path):
    with open(input_path, 'r') as file:
        lines = file.readlines()
    
    processed_lines = []
    
    for line in lines:
        # Remove Rotation part
        line = re.sub(r'Rotation: P=[^ ]+ Y=[^ ]+ R=[^ ]+', '', line)
        # Extract only the numbers and keep the first 6 sets
        numbers = re.findall(r'-?\d+\.?\d*', line)[:6]
        if numbers:
            processed_lines.append(" ".join(numbers))
    
    with open(output_path, 'w') as file:
        file.write("\n".join(processed_lines))

def convert_ring_pass_list_to_number_list(str_list):
    converted_list = [[int(item.strip()) for item in sublist] for sublist in str_list]
    return converted_list

def convert_ring_location_list_to_number_array(string_list):
    flat_list = [
        [float(num) for num in line.split()]
        for sublist in string_list for line in sublist if line.strip()
    ]
    
    # Convert to numpy array and reshape to the desired shape
    array = np.array(flat_list)
    reshaped_array = array.reshape((-1, 15, 6))
    
    return reshaped_array
def combine_ring_pass_and_ring_loc(numb_ring_passed_each_trial, ring_location_arr):
    ring_loc_lst = []
    for i in range(len(numb_ring_passed_each_trial)):
        ring_loc_lst.append(ring_location_arr[5+i,:numb_ring_passed_each_trial[i],:3].tolist())
    return ring_loc_lst

def process_ring_location_txt(ring_pass_lst, ring_location_lst):
    ring_pass_lst = convert_ring_pass_list_to_number_list(ring_pass_lst)
    ring_location_arr = convert_ring_location_list_to_number_array(ring_location_lst)

    if len(ring_pass_lst) >= 50:
        # have 50 trials, 5 caliberation at beginning
        numb_ring_passed_each_trial = [len(i) for i in ring_pass_lst]
        numb_ring_passed_each_trial = numb_ring_passed_each_trial[5:50]
        ring_loc_lst = combine_ring_pass_and_ring_loc(numb_ring_passed_each_trial, ring_location_arr)
    elif len(ring_pass_lst) == 35:
        # have 35 trials, 5 caliberation at beginning
        numb_ring_passed_each_trial = [len(i) for i in ring_pass_lst]
        numb_ring_passed_each_trial = numb_ring_passed_each_trial[5:35]
        ring_loc_lst = combine_ring_pass_and_ring_loc(numb_ring_passed_each_trial, ring_location_arr)
    else:
        print("wrong -- please double check if you have enough data")
    return ring_loc_lst

def sort_txt_file_names(filename):
    match = re.search(r'PassRing_(\d+).txt', filename)
    return int(match.group(1)) if match else float('inf')

def process_communication_txt(communication_txt, len_trials):
    # Define the replacements
    replacements = {'no': 'No', 'word': 'Word', 'full': 'Free'}
    

    # Function to perform replacements
    def replace_words(text):
        for key, value in replacements.items():
            text = text.replace(key, value)
        return text
    
    # Process the list
    processed_list = [replace_words(item).strip() for item in communication_txt]
    duplicated_list = []

    if len_trials >= 50:
        # Split processed elements into individual words and duplicate each word 5 times
        for item in processed_list:
            words = item.split()
            for word in words:
                duplicated_list.extend([word] * 5)

    elif len_trials == 35:
        for item in processed_list[:-1]:
            words = item.split()
            for word in words:
                duplicated_list.extend([word] * 5)
    return duplicated_list

def ring_comm_lst_to_df(team_session_lst, ring_loc_lst, communication_lst):
    team_lst = []
    session_lst = []
    trial_lst = []
    ring_lst = []
    ring_x_loc_lst = []
    ring_y_loc_lst = []
    ring_z_loc_lst = []
    comm_lst = []
    diff_lst = []

    # for each team each session
    for team_sess in range(len(team_session_lst)):
        # for each trial
        for trial_id in range(len(ring_loc_lst[team_sess])):
            # for each ring
            if len(ring_loc_lst[team_sess][trial_id]) == 0:
                team_lst.append(team_session_lst[team_sess][0])
                session_lst.append(team_session_lst[team_sess][1])
                trial_lst.append(trial_id)
                ring_lst.append(0)
                ring_x_loc_lst.append(0)
                ring_y_loc_lst.append(0)
                ring_z_loc_lst.append(0)
                comm_lst.append(communication_lst[team_sess][trial_id])
                diff_lst.append('Easy')
                continue
            for ring_id in range(len(ring_loc_lst[team_sess][trial_id])):
                team_lst.append(team_session_lst[team_sess][0])
                session_lst.append(team_session_lst[team_sess][1])
                trial_lst.append(trial_id)
                ring_lst.append(ring_id)
                ring_x_loc_lst.append(ring_loc_lst[team_sess][trial_id][ring_id][0])
                ring_y_loc_lst.append(ring_loc_lst[team_sess][trial_id][ring_id][1])
                ring_z_loc_lst.append(ring_loc_lst[team_sess][trial_id][ring_id][2])
                comm_lst.append(communication_lst[team_sess][trial_id])
                if ring_id < 5:
                    diff_lst.append('Easy')
                elif ring_id < 10 :
                    diff_lst.append('Medium')
                else:
                    diff_lst.append('Hard')

    output_df = pd.DataFrame({'teamID': team_lst,
                  'sessionID': session_lst,
                  'trialID': trial_lst,
                  'ringID': ring_lst,
                  'communication': comm_lst,
                  'difficulty': diff_lst,
                  'ringX': ring_x_loc_lst,
                  'ringY': ring_y_loc_lst,
                  'ringZ': ring_z_loc_lst})
    return output_df


def change_sample_rate(data, time, target_sample_rate, resample_axis=0):
    trial_time_length = time[-1]-time[0]
    resampled_data = signal.resample(data, round(target_sample_rate * trial_time_length), time, axis=resample_axis, window=4)
    return resampled_data

def downsample_location_df(raw_location):
    downsampled_location = copy.deepcopy(raw_location)
    downsampled_location = downsampled_location.drop(columns={'location', 'time'})

    downsampled_location['location'] = None
    downsampled_location['time'] = None
    for i in tqdm(range(len(downsampled_location))):
        resampled_loc, resampled_time = change_sample_rate(raw_location.iloc[i].location, raw_location.iloc[i].time, 60,1)
        downsampled_location.at[i,'location'] = resampled_loc
        downsampled_location.at[i,'time'] = resampled_time

    return downsampled_location

def find_trial_indices(data, min_length=90):
    start_indices = []
    end_indices = []
    in_trial = False
    trial_start = 0

    for i in range(1, len(data)):
        if not in_trial:
            if data[i] > data[i-1]:
                trial_start = i-1
                in_trial = True
        else:
            if data[i] <= data[i-1] or i == len(data) - 1:
                trial_end = i if i == len(data) - 1 else i - 1
                if trial_end - trial_start + 1 >= min_length:
                    start_indices.append(trial_start)
                    end_indices.append(trial_end)
                in_trial = False

    return start_indices, end_indices


def check_and_remove_in_trial_breaks(location, start_idx, end_idx):
    zero_idxs = np.where(location == -1)[0]

    trial_id = 0
    bad_trial_idx_lst = []

    for i in range(1,len(zero_idxs)-1):
        location_diff = location[zero_idxs[i]+1] - location[zero_idxs[i]-1]
        if location_diff > 0:
            if location_diff > 45000:
                trial_id += 1
            else:
                bad_trial_idx_lst.append(trial_id)
                trial_id += 1

    for index in sorted(bad_trial_idx_lst, reverse=True):
        del start_idx[index]
        del end_idx[index]
    return start_idx, end_idx


def append_to_trial_location_df(start_idx, end_idx, team_sess, temp_location, temp_time, trialed_location, role):
    for j in range(len(start_idx)):
        temp_trial_idx = trialed_location.loc[(trialed_location.teamID == team_sess.teamID) & 
                         (trialed_location.sessionID == team_sess.sessionID) & 
                         (trialed_location.trialID == j)].index[0]
        trialed_location.at[temp_trial_idx, f'{role.lower()}Location'] = temp_location[:,start_idx[j]:end_idx[j]]
        trialed_location.at[temp_trial_idx, f'{role.lower()}Time'] = temp_time[start_idx[j]:end_idx[j]]

def match_ring_passed_to_trials(start_idx, end_idx, temp_ring_epochs, locationX, numb_tirals):
    return start_idx[:numb_tirals], end_idx[:numb_tirals]



def trial_location(raw_location, ring_comm_diff_df):
    trialed_location = copy.deepcopy(ring_comm_diff_df)
    trialed_location = trialed_location.drop(columns={'ringID', 'difficulty', 'ringX', 'ringY', 'ringZ'}).drop_duplicates().reset_index(drop=True)
    trialed_location['yawLocation'] = None
    trialed_location['yawTime'] = None
    trialed_location['pitchLocation'] = None
    trialed_location['pitchTime'] = None
    trialed_location['thrustLocation'] = None
    trialed_location['thrustTime'] = None
    unique_team_session = raw_location[['teamID', 'sessionID']].drop_duplicates()
    for role in ['Yaw', 'Pitch', 'Thrust']:
        for i, team_sess in unique_team_session.iterrows():
            temp_location = raw_location.loc[(raw_location.role == role) &(raw_location.teamID == team_sess.teamID) & (raw_location.sessionID == team_sess.sessionID)]['data'].iloc[0].T
            temp_time = raw_location.loc[(raw_location.role == role) &(raw_location.teamID == team_sess.teamID) & (raw_location.sessionID == team_sess.sessionID)]['time'].iloc[0]
            temp_time = temp_time - temp_time[0]
            start_idx,end_idx = find_trial_indices(temp_location[0,:])
            temp_ring_epochs = ring_comm_diff_df.loc[(ring_comm_diff_df.teamID == team_sess.teamID) & (ring_comm_diff_df.sessionID == team_sess.sessionID)]
            temp_numb_trials = len(temp_ring_epochs.trialID.unique())
            if len(start_idx) == temp_numb_trials:
                # same number of trials found in location.
                append_to_trial_location_df(start_idx, end_idx, team_sess, temp_location, temp_time, trialed_location, role)
            elif len(start_idx) > temp_numb_trials:
                # more trial found in location than epoched rings
                # check if 45th trial is the last trial
                # T10S2 have problem here
                start_idx, end_idx = check_and_remove_in_trial_breaks(temp_location[0], start_idx,end_idx)
                if len(start_idx) == temp_numb_trials:
                    # same number of trials found in location now, directly trial locations
                    append_to_trial_location_df(start_idx, end_idx, team_sess, temp_location, temp_time, trialed_location, role)
                else:
                    # still contain extra trials, need to check nubmer ring passed in each trial as reference
                    start_idx, end_idx = match_ring_passed_to_trials(start_idx, end_idx, temp_ring_epochs, temp_location[0], temp_numb_trials)
                    append_to_trial_location_df(start_idx, end_idx, team_sess, temp_location, temp_time, trialed_location, role)
            else:
                # less trial found in location than epoched rings
                print('something is wrong here')
                append_to_trial_location_df(start_idx, end_idx, team_sess, temp_location, temp_time, trialed_location, role)

    return trialed_location



def read_corresponding_ring_communication(raw_location):
    # use thrust location as the location fo the team
    raw_location = raw_location[raw_location.role == 'Thrust'].reset_index(drop=True)
    raw_location = raw_location.drop(columns={'modality', 'desktop', 'role'})
    ring_loc_lst = []
    communication_lst = []
    team_session_lst = []
    numb_ring_passed_lst = []
    for i in range(len(raw_location)):
        trial_numb_pass_lst = []
        trial_ring_pos_lst = []
        team_session = raw_location.iloc[i][['teamID', 'sessionID']]
        ring_pass_file_lst = glob.glob(f'{path}/{team_session.teamID}_{team_session.sessionID}/PassRing_*.txt')
        ring_location_file_lst = glob.glob(f'{path}/{team_session.teamID}_{team_session.sessionID}/RingPositions_*.txt')

        for file in sorted(ring_pass_file_lst, key=sort_txt_file_names):
            ring_pass = read_txt(file)
            trial_numb_pass_lst.append(ring_pass)
        for file in sorted(ring_location_file_lst, key=sort_txt_file_names):
            ring_pos = read_txt(file)
            trial_ring_pos_lst.append(ring_pos)
    
        comm = read_txt(f'{path}/{team_session.teamID}_{team_session.sessionID}/CommMethod.txt')
        ring_locations = process_ring_location_txt(trial_numb_pass_lst, trial_ring_pos_lst)
        ring_loc_lst.append(ring_locations)
        communication_lst.append(process_communication_txt(comm, len(trial_numb_pass_lst)))
        team_session_lst.append([team_session.teamID, team_session.sessionID])
    return ring_comm_lst_to_df(team_session_lst, ring_loc_lst, communication_lst)

def epoch_location_df(location_df, ring_comm_diff_df):
    ring_comm_diff_df['location'] = None
    ring_comm_diff_df['time'] = None
    for i in tqdm(range(len(ring_comm_diff_df))):
        temp_team_sess_trial = ring_comm_diff_df.iloc[i][['teamID', 'sessionID', 'trialID']]
        team_loc = location_df.loc[(location_df.teamID == temp_team_sess_trial.teamID)
                       &(location_df.sessionID == temp_team_sess_trial.sessionID)
                       &(location_df.trialID == temp_team_sess_trial.trialID)].iloc[0]
        ring_idx = np.where(team_loc.location[0] <= ring_comm_diff_df.iloc[i].ringX)[0]
        if len(ring_idx) > 0:
            ring_idx = ring_idx[-1]
        else:
            # recording starts after trial start. Ignore and continue
            continue
        if ring_idx - 90 > 0 and ring_idx + 90 <= len(team_loc.location[0]):
            ring_comm_diff_df.at[i, 'location'] = team_loc.location[:, ring_idx-90:ring_idx+90]
            ring_comm_diff_df.at[i, 'time'] = team_loc.time[ring_idx-90:ring_idx+90]

        else:
            # ignore this ring and continue
            continue
    return ring_comm_diff_df.dropna().reset_index(drop=True)

def epoch_location_without_downsample(location_df, ring_comm_diff_df):
    
    epoched_location_df = copy.deepcopy(ring_comm_diff_df)
    epoched_location_df['location'] = None
    epoched_location_df['time'] = None
    epoched_location_df['startTime'] = None
    epoched_location_df['ringTime'] = None
    for i in tqdm(range(len(ring_comm_diff_df))):
        temp_team_sess_trial = ring_comm_diff_df.iloc[i][['teamID', 'sessionID', 'trialID']]
        team_loc = location_df.loc[(location_df.teamID == temp_team_sess_trial.teamID)
                       &(location_df.sessionID == temp_team_sess_trial.sessionID)
                       &(location_df.trialID == temp_team_sess_trial.trialID)].iloc[0]
        start_idx = np.where(team_loc.location[0] >= ring_comm_diff_df.iloc[i].ringX-50000)[0]
        end_idx = np.where(team_loc.location[0] >= ring_comm_diff_df.iloc[i].ringX+50000)[0]
        ring_idx = np.where(team_loc.location[0] >= ring_comm_diff_df.iloc[i].ringX)[0]
        if len(start_idx) > 0 and len(end_idx) > 0:
            start_idx = start_idx[0]
            end_idx = end_idx[0]
            ring_idx = ring_idx[0]
            if ring_idx <= start_idx + 80 or ring_idx >= start_idx+480:
                continue
        else:
            # Ignore and continue
            continue
        epoched_location_df.at[i, 'location'] = team_loc.location[:, start_idx:end_idx]
        epoched_location_df.at[i, 'time'] = team_loc.time[start_idx:end_idx]
        epoched_location_df.at[i, 'startTime'] = team_loc.time[start_idx]
        epoched_location_df.at[i, 'ringTime'] = team_loc.time[ring_idx]
    return epoched_location_df.dropna().reset_index(drop=True)


if __name__ == '__main__':
    # load data
    path = '../../data/raw_xdf'
    pd.set_option('display.max_columns', None)
    downsample_epoch = False
    # raw_eeg = pd.DataFrame()
    # raw_location = pd.DataFrame()
    # raw_audio = pd.DataFrame()
    # raw_pupilsize = pd.DataFrame()
    # raw_openness = pd.DataFrame()

    # for file_name in tqdm(os.listdir(path)):
    #     if not file_name.startswith('.'):
    #         print(file_name)
    #         if int(file_name.split("_")[0][1:]) < 14:
    #             continue
    #         full_path = os.path.join(path, file_name)
    #         target_file = glob.glob(os.path.join(full_path, '*.xdf'))
    #         if len(target_file) > 1:
    #             all_data = merge_xdf(target_file, file_name.split('_'))
    #         else:
    #             try:
    #                 temp_xdf = read_xdf(target_file[0])
    #                 all_data = remove_cache_data(temp_xdf, file_name.split('_'))
    #             except:
    #                 print('error exist in')
    #                 print(file_name)
    #         raw_eeg = pd.concat([raw_eeg, all_data.loc[all_data['modality'] == 'EEG']])
    #         raw_location = pd.concat([raw_location, all_data.loc[all_data['modality'] == 'Location']])
    #         raw_audio = pd.concat([raw_audio, all_data.loc[all_data['modality'] == 'Audio']])
    #         raw_pupilsize = pd.concat([raw_pupilsize, all_data.loc[all_data['modality'] == 'PupilSize']])
    #         raw_openness = pd.concat([raw_openness, all_data.loc[all_data['modality'] == 'Openness']])

    # raw_eeg = sort_data_frame(raw_eeg)
    # raw_location = sort_data_frame(raw_location)
    # raw_audio = sort_data_frame(raw_audio)
    # raw_pupilsize = sort_data_frame(raw_pupilsize)
    # raw_openness = sort_data_frame(raw_openness)

    # raw_eeg.to_pickle(os.path.join(path, '../', 'raw_eeg.pkl' ))
    # raw_location.to_pickle(os.path.join(path, '../', 'raw_location.pkl' ))
    # raw_audio.to_pickle(os.path.join(path, '../', 'raw_audio.pkl' ))
    # raw_pupilsize.to_pickle(os.path.join(path, '../', 'raw_pupilsize.pkl' ))
    # raw_openness.to_pickle(os.path.join(path, '../', 'raw_openness.pkl' ))

    raw_location = pd.read_pickle(os.path.join(path, '../raw_pickle', 'raw_location.pkl' ))
    ring_comm_diff_df = read_corresponding_ring_communication(raw_location)
    ring_comm_diff_df.to_pickle(os.path.join(path, '../', 'team_performance.pkl' ))

    trialed_location = trial_location(raw_location, ring_comm_diff_df)
    trialed_location.to_pickle(os.path.join(path, '../', 'trialed_location.pkl' ))
    if downsample_epoch:
        downsampled_location = downsample_location_df(trialed_location)
        epoch_location = epoch_location_df(downsampled_location, ring_comm_diff_df)
        epoch_location.to_pickle(os.path.join(path, '../', 'epoched_downsampled_location.pkl' ))
    else:
        epoch_location = epoch_location_without_downsample(trialed_location, ring_comm_diff_df)
        epoch_location.to_pickle(os.path.join(path, '../', 'epoched_raw_location.pkl' ))



