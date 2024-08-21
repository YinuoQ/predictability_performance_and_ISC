import os
import sys
import copy
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
sys.path.insert(1, '../preprocessing/')
from performance_from_location import get_performance_from_location

def get_performance(lcoation_df):
    performance_df = get_performance_from_location(lcoation_df)
    return performance_df
def compute_predictability(target_prediction_arr):
    import IPython
    IPython.embed()
    assert False
if __name__ == '__main__':
    path = '../../data'
    pd.set_option('display.max_columns', None)
    lcoation_df = pd.read_pickle(os.path.join(path, 'epoched_raw_location.pkl'))
    target_prediction_arr = np.read('transformer/log_data_seed_1/lightning_logs/version_0/pred_target.npy')


    # epoch based performances
    performance_df = get_performance(lcoation_df)
    predictability = compute_predictability(target_prediction_arr)
    # mixed_effects_model(speech_ISC_df)
   
    # # trial based performances
    # speech_performance_df = get_trialed_speech_ISC_with_performance(lcoation_df, speech_event)
    # mixed_effects_model(speech_performance_df)




