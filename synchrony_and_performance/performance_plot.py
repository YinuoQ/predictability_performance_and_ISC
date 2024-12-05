import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM 
sys.path.insert(1, 'preprocessing/')
from performance_from_location import get_performance_from_location

def get_performance(lcoation_df):
    performance_df = get_performance_from_location(lcoation_df)
    performance_df = performance_df.drop(columns=['ringX', 'ringY', 'ringZ', 'location', 'time',
       'startTime', 'ringTime'])
    return performance_df

def plot_team_total_performance(performance_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = performance_df.groupby(['sessionID', 'teamID']).apply(lambda x: len(x))
    averaged_y = y.groupby('sessionID').mean()
    team_y = y.unstack(level='teamID').values
    ax.bar(x, averaged_y, color='#7D8598')
    ax.plot(np.array([x]*18).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0,599])
    ax.set_yticks([0, 100, 200, 300, 400, 500])
    ax.set_yticklabels([0, 100, 200, 300, 400, 500], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)

    # repeated t-test

    y_df = y.reset_index(name='performance')
    y_df = y_df[y_df.teamID != 'T7'].reset_index(drop=True)
    results = print(AnovaRM(data=y_df, depvar='performance', 
              subject='teamID', within=['sessionID']).fit())


    plt.savefig('../plots/performance_session.png', dpi=300)

def plot_team_global_performance(performance_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = performance_df.groupby(['sessionID', 'teamID', 'trialID']).apply(lambda x: len(x))
    averaged_y = y.groupby(['sessionID', 'teamID']).mean().groupby(['sessionID']).mean()
    team_y = y.groupby(['sessionID', 'teamID']).mean().unstack(level='teamID').values
    ax.bar(x, averaged_y, color='#7D8598')
    ax.plot(np.array([x]*18).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0,12])
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.set_yticklabels([0, 2, 4, 6, 8, 10], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)
    # repeated t-test
    y_df = y.groupby(['sessionID', 'teamID']).mean().reset_index(name='performance')
    y_df = y_df[y_df.teamID != 'T7'].reset_index(drop=True)
    results = print(AnovaRM(data=y_df, depvar='performance', 
              subject='teamID', within=['sessionID']).fit())
    # import IPython
    # IPython.embed()
    # assert False
    plt.savefig('../plots/performance_global.png', dpi=300)

def plot_team_local_performance(performance_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = performance_df.groupby(['sessionID', 'teamID']).apply(lambda x: x.performance.mean())
    averaged_y = y.groupby('sessionID').mean()
    team_y = y.unstack(level='teamID').values
    ax.bar(x, averaged_y, color='#7D8598')
    ax.plot(np.array([x]*18).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0.6,1])
    ax.set_yticks([.6, .7, .8, .9])
    ax.set_yticklabels([.6, .7, .8, .9], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)
    # repeated t-test
    y_df = y.reset_index(name='performance')
    y_df = y_df[y_df.teamID != 'T7'].reset_index(drop=True)
    results = print(AnovaRM(data=y_df, depvar='performance', 
              subject='teamID', within=['sessionID']).fit())

    plt.savefig('../plots/performance_local.png', dpi=300)

def main():
    path = '../data'
    pd.set_option('display.max_columns', None)
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    performance_df = get_performance(lcoation_df)

    plot_team_total_performance(performance_df)
    plot_team_global_performance(performance_df)
    plot_team_local_performance(performance_df)

if __name__ == '__main__':
    main()







