import os
import sys
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from statsmodels.stats.anova import AnovaRM 

def get_helpfulness_familiarity(path):
    helpfulness_df = pd.read_csv(os.path.join(path, 'helpfulness.csv'))
    familiarity_df = pd.read_csv(os.path.join(path, 'familiarity.csv'))

    return helpfulness_df, familiarity_df

def plot_helpfulness(helpfulness_df):

    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = helpfulness_df.groupby(['session']).apply(lambda x: x.helpfulness.mean())
    team_y = helpfulness_df.groupby(['team']).apply(lambda x: x.helpfulness)
    team_y = np.reshape(team_y.values, [3, -1], order='F')
    ax.bar(x, y, color='#BBE2EA')
    ax.plot(np.array([x]*18).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0,15])
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12, 14], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)
    # repeated t-test
    results = print(AnovaRM(data=helpfulness_df, depvar='helpfulness', 
              subject='team', within=['session']).fit())

    plt.savefig('../plots/helpfulness.png', dpi=300)

def plot_familarity(familiarity_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = familiarity_df.groupby(['session']).apply(lambda x: x.familiarity.mean())
    team_y = familiarity_df.groupby(['team']).apply(lambda x: x.familiarity)
    team_y = np.reshape(team_y.values, [3, -1], order='F')
    ax.bar(x, y, color='#BBE2EA')
    ax.plot(np.array([x]*18).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0,15])
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.set_yticklabels([0, 2, 4, 6, 8, 10, 12, 14], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)
    # repeated t-test
    results = print(AnovaRM(data=familiarity_df, depvar='familiarity', 
              subject='team', within=['session']).fit())

    plt.savefig('../plots/familiarity.png', dpi=300)

def main():
    path = '../data'
    pd.set_option('display.max_columns', None)
    helpfulness_df, familiarity_df = get_helpfulness_familiarity(path)

    plot_helpfulness(helpfulness_df)
    plot_familarity(familiarity_df)

if __name__ == '__main__':
    main()







