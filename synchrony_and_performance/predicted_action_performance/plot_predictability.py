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


def plot_predictability(predictability_df):
    fig, ax = plt.subplots(1,1, figsize=(3, 4))
    x = np.arange(3)
    y = predictability_df.groupby(['sessionID', 'teamID']).apply(lambda x: np.mean(x.predictability))
    averaged_y = y.groupby(['sessionID']).mean()
    team_y = y.unstack(level='teamID').values
    ax.bar(x, averaged_y, color='#BEEAAF')
    ax.plot(np.array([x]*11).T, team_y, '-o', color='k', alpha=0.3,markeredgecolor='none', markersize=5)
    ax.set_xticks(x, ['Session 1', 'Session 2', 'Session 3'], font='helvetica', fontsize=12)
    ax.set_ylim([0.1,0.7])
    ax.set_yticks([0.2, .4, .6])
    ax.set_yticklabels([0.2, .4, .6], font='helvetica', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    print(team_y)
    # repeated t-test
    y_df = y.reset_index(name='performance')

    session_counts = y_df.groupby('teamID')['sessionID'].nunique()
    balanced_team_ids = session_counts[session_counts == y_df['sessionID'].nunique()].index
    # Filter the original DataFrame to keep only balanced teamIDs
    balanced_df = y_df[y_df['teamID'].isin(balanced_team_ids)]
    results = print(AnovaRM(data=balanced_df, depvar='performance', 
              subject='teamID', within=['sessionID']).fit())
    plt.savefig('../../plots/predictability.png', dpi=300)


if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    predictability_df = pd.DataFrame()
    for seed in [1,2,3,4]:
        temp_pred = get_predictability(seed)
        predictability_df = pd.concat((predictability_df, temp_pred))
    predictability_df = predictability_df.reset_index(drop=True)
    plot_predictability(predictability_df)




