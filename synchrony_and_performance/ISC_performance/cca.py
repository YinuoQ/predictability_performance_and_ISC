import copy
import numpy as np
import pandas as pd
from numpy.linalg import eig
from tqdm import tqdm
from timeit import default_timer
import matplotlib.pyplot as plt
from scipy import linalg as sp_linalg
from scipy.sparse import diags

def regInv(R, k):
    '''PCA regularized inverse of square symmetric positive definite matrix R.'''

    U, S, Vh = np.linalg.svd(R)
    invR = U[:, :k].dot(diags(1 / S[:k]).toarray()).dot(Vh[:k, :])
    return invR
def ISC_eeg(data):
    gamma = 0.1# shrinkage parameter; smaller gamma for less regularization
    if type(data) != list and  type(data) != np.ndarray:
        X = list(data.values())
        if type(X[0]) == dict:
            X_new = []
            for idx in range(len(X)):
                X_new+=X[idx].values()
            X = X_new
        if type(X[0]) == list:
            X_new = []
            for idx in range(len(X)):
                X_new.append(np.array(X[idx]).reshape(3,20,-1))
            X = X_new
    else:
        X = [data]

    # N subjects, D channels, T samples
    N, D, data_time = np.shape(data)
    temp = []
    for n in range(N):
        temp.append(np.cov(data[n]))

    Rw = np.nansum(temp, axis=0)
    Rt = N**2 * np.cov(np.nanmean(data, axis=0))
    Rb = (Rt - Rw) / (N-1)

    rank = np.linalg.matrix_rank(Rw)
    k = min(D, rank)

    if k < D:
        invR = regInv(Rw, k)
        ISC, W = sp_linalg.eig(invR.dot(Rb))
        ISC, W = ISC[:k], W[:, :k]


    else:
        Rw_reg = (1-gamma) * Rw + gamma * Rw.diagonal().mean() * np.identity(D)
        ISC, W = sp_linalg.eig(Rb, Rw_reg) # W is already sorted by eigenvalue and normalized

    ISC = np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W))
    ISC, W = np.real(ISC), np.real(W)        

    return np.real(ISC), np.real(W)
def comput_ISC(W, data):
    N, D, T = data.shape

    Rw = sum(np.cov(data[n,...]) for n in range(N))
    Rt = N**2 * np.cov(data.mean(axis=0))
    Rb = (Rt - Rw) / (N-1)

    ISC = np.sort(np.diagonal(W.T.dot(Rb).dot(W)) / np.diag(W.T.dot(Rw).dot(W)))[::-1]

    return np.real(ISC)


def window(data, window_size=64, overlap=32):
    if type(data) == list:
        data_arr = np.array(data)
        reshapred_data = np.transpose(data_arr, [1,2,0,3])
        reshapred_data = reshapred_data.reshape(3,20,-1)
        data = reshapred_data
    view = np.lib.stride_tricks.sliding_window_view(data, window_shape=window_size,axis=2)[:,:,0::int(window_size-overlap),:]
    return view.copy()

def ISC_sliding_window(W, data, window_size=32, overlap=16):
    data_slidng_window = window(data, window_size=window_size, overlap=overlap)
    time_related_ISC_lst = []

    for i in range(data_slidng_window.shape[2]):
        temp_ISC = comput_ISC(W, data_slidng_window[:,:,i,:])[0]
        time_related_ISC_lst.append(temp_ISC)
    return np.array(time_related_ISC_lst)

















