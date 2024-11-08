import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM 
from eeg_ISC_performance import compute_eeg_ISC
from pupil_ISC_performance import compute_pupil_ISC
from action_ISC_performance import compute_action_ISC
from speech_event_ISC_performance import compute_speech_ISC



def plot_eeg_ISC(eeg_isc_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = eeg_isc_df.groupby(['sessionID', 'teamID']).apply(lambda x: x.eegISC.mean())
    averaged_y = y.groupby('sessionID').mean()
    team_y = y.unstack(level='teamID').values
    ax.bar(x, averaged_y, color='#EDA39C')
    ax.plot(np.array([x]*11).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0,1])
    ax.set_yticks([0, .2, .4, .6, .8])
    ax.set_yticklabels([0, .2, .4, .6, .8], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)

    # repeated t-test

    y_df = y.reset_index(name='performance')
    y_df = y_df[y_df.teamID != 'T14'].reset_index(drop=True)
    results = print(AnovaRM(data=y_df, depvar='performance', 
              subject='teamID', within=['sessionID']).fit())

    plt.savefig('../../plots/synchrony_eeg.png', dpi=300)

def plot_pupil_ISC(pupil_isc_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = pupil_isc_df.groupby(['sessionID', 'teamID']).apply(lambda x: x.pupilISC.mean())
    averaged_y = y.groupby('sessionID').mean()
    team_y = y.unstack(level='teamID').values
    ax.bar(x, averaged_y, color='#D5EAFA')
    ax.plot(np.array([x]*18).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0,1])
    ax.set_yticks([0, .2, .4, .6, .8])
    ax.set_yticklabels([0, .2, .4, .6, .8], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)

    plt.savefig('../../plots/synchrony_pupil.png', dpi=300)

def plot_action_ISC(action_isc_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = action_isc_df.groupby(['sessionID', 'teamID']).apply(lambda x: x.actionISC.mean())
    averaged_y = y.groupby('sessionID').mean()
    team_y = y.unstack(level='teamID').values
    ax.bar(x, averaged_y, color='#AC8FCC')
    ax.plot(np.array([x]*18).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0,0.7])
    ax.set_yticks([0, .2, .4, .6])
    ax.set_yticklabels([0, .2, .4, .6], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)

    plt.savefig('../../plots/synchrony_action.png', dpi=300)

def plot_speech_ISC(speech_isc_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = speech_isc_df.groupby(['sessionID', 'teamID']).apply(lambda x: x.speechISC.mean())
    averaged_y = y.groupby('sessionID').mean()
    team_y = y.unstack(level='teamID').values
    ax.bar(x, averaged_y, color='#F8D094')
    ax.plot(np.array([x]*17).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0,1.1])
    ax.set_yticks([0, .2, .4, .6, .8, 1])
    ax.set_yticklabels([0, .2, .4, .6, .8, 1], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)

    plt.savefig('../../plots/synchrony_speech.png', dpi=300)


def main():
    path = '../../data'
    # pd.set_option('display.max_columns', None)
    # eeg_isc_df = compute_eeg_ISC(pd.read_pickle(os.path.join(path, 'epoched_EEG.pkl')))
    # eeg_isc_df = eeg_isc_df.drop(columns=['yawEEG', 'pitchEEG', 'thrustEEG'])
    # pupil_isc_df = compute_pupil_ISC(pd.read_pickle(os.path.join(path, 'epoched_pupil.pkl')))
    # pupil_isc_df = pupil_isc_df.drop(columns=['yawPupil', 'pitchPupil', 'thrustPupil', 'ringTime'])
    action_isc_df = compute_action_ISC(pd.read_pickle(os.path.join(path, 'epoched_action.pkl')))
    action_isc_df = action_isc_df.drop(columns=['yawAction', 'pitchAction', 'thrustAction'])
    # speech_isc_df = pd.read_pickle(os.path.join(path, 'epoched_speech_event.pkl'))
    # speech_isc_df = speech_isc_df[speech_isc_df.communication != 'No'].reset_index(drop=True)
    # speech_isc_df = speech_isc_df.drop(speech_isc_df[(speech_isc_df['teamID'] == 'T8') & (speech_isc_df['sessionID'] == 'S2')].index)
    # speech_isc_df = speech_isc_df.drop(speech_isc_df[(speech_isc_df['teamID'] == 'T21') & (speech_isc_df['sessionID'] == 'S1')].index)
    # speech_isc_df = compute_speech_ISC(speech_isc_df)
    # speech_isc_df = speech_isc_df.drop(columns=['yawSpeech', 'pitchSpeech', 'thrustSpeech'])

    # plot_eeg_ISC(eeg_isc_df)
    # plot_pupil_ISC(pupil_isc_df)
    plot_action_ISC(action_isc_df)
    # plot_speech_ISC(speech_isc_df)

if __name__ == '__main__':
    main()







