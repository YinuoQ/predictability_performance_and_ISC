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

def get_performance(lcoation_df):
    performance_df = get_performance_from_location(lcoation_df)
    performance_df = performance_df.drop(columns=['ringX', 'ringY', 'ringZ', 'location', 'time',
       'startTime', 'ringTime'])
    return performance_df

def plot_team_performance_hist(performance_df):

    fig, ax = plt.subplots(1,1, figsize=(6, 4))
    x = np.arange(3)
    y = performance_df.groupby(['sessionID', 'teamID', 'trialID']).apply(lambda x: len(x))
    team_y = y.values
   
    team_y_percentage = team_y / np.nansum(team_y, axis=0) * 100  # Normalize each session to 100%


    import IPython
    IPython.embed()
    assert False


    plt.hist(team_y)
    plt.xlabel("Performance")
    plt.ylabel("Trials")
    plt.xticks(list(range(1,16)))
    # plt.legend(["Session 1", "Session 2", "Session 3"])
    plt.savefig('../plots/performance_hist.png', dpi=300)
    plt.close()


def main():
    path = '../data'
    pd.set_option('display.max_columns', None)
    performance_df = pd.read_pickle(os.path.join(path, 'team_performance.pkl'))

    plot_team_performance_hist(performance_df)


if __name__ == '__main__':
    main()







