import os
import sys
import glob
import yaml
import copy
import torch
import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial import distance
from scipy.signal import correlate
from torchmetrics.classification import MulticlassAUROC
from utils import common
from munch import munchify
from collections import OrderedDict
from models import ActionPredictionModel
from pytorch_lightning.plugins import DDPPlugin
from dataset import PredictAction
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from collections import OrderedDict

def load_config(filepath):
    with open(filepath, 'r') as stream:
        try:
            trainer_params = yaml.safe_load(stream)
            return trainer_params
        except yaml.YAMLError as exc:
            print(exc)

def seed(cfg):
    torch.manual_seed(cfg.seed)
    if cfg.if_cuda:
        torch.cuda.manual_seed(cfg.seed)

def pearson_correlation(datax, datay):
    numerator1 = datax - np.nanmean(datax)
    numerator2 = datay - np.nanmean(datay)
    numerator = np.nansum(numerator1*numerator2)
    denometor1 = np.nansum((datax - np.nanmean(datax))**2)
    denometor2 = np.nansum((datay - np.nanmean(datay))**2)
    denometor = np.sqrt(denometor1*denometor2)
    return np.abs(numerator / denometor)
def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 

    Returns
    ----------
    crosscorr : float
    """
    datay_shift = copy.deepcopy(datay)
    datay_shift[:lag] = np.nan
    datay_shift[lag:] = datay[lag:]
    
    return pearson_correlation(datax, datay_shift)

def nanargmax(a, axis=0):
    arg_max_idx_lst = []
    for i in range(a.shape[0]):
        if np.sum(np.isnan(a[i])) == a[i].shape[0]:
            arg_max_idx_lst.append(np.nan)
        else:
            arg_max_idx_lst.append(np.nanargmax(a[i]))
    return np.array(arg_max_idx_lst)

def evaluation_matrics(pred_output, target, transformer):
    # auroc = MulticlassAUROC(num_classes=3)

    pred_stacked = torch.stack(pred_output).permute(1,2,0)

    target = target.to('cpu')
    pred_stacked = pred_stacked.to('cpu')

    # prediction results after classification
    pred_results = []
    for acc_idx in range(len(pred_output)):
        pred_results.append(torch.argmax(pred_output[acc_idx], axis=1).tolist())
    pred_results = torch.tensor(pred_results).T
    pred_results[pred_results == 2] = -1
    target[target == 2] = -1

    pred_results_arr = pred_results.numpy()
    target_arr = target.numpy()
    r = np.corrcoef(pred_results_arr.flatten(), target_arr.flatten())[0,1]
    return  r, pred_results_arr, target_arr


def one_hot_encoding(data):
    # one-hot encoding for nn-results
    data_arr = np.array(data)
    data_tensor = torch.tensor((np.arange(3) == data_arr[...,None]).astype(int))    
    return data_tensor.permute(1, 0, 2).float()


def test_Transformer(test_loader, model, log_dir):
    target_lst = []
    prediction_results_lst = []
    model_performance_lst = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for batch_idx, data in enumerate(test_loader):
        target = data[2]
        gpu_data = []
        for p_data in data:
            gpu_data.append(p_data.to(device))
        
        _, pred_output = model.test_step(gpu_data, batch_idx)
        r, prediction, target = evaluation_matrics(list(pred_output.permute(1, 0, 2)), target.to(device), True)
        target_lst.append(target)
        prediction_results_lst.append(prediction)
        model_performance_lst.append(r)

    print('######################################################################################')
    print(np.tanh(np.mean(np.arctanh(model_performance_lst))))
    print('######################################################################################')
    
    save_path = os.path.join(*log_dir.split('/')[:3])
    target_prediction_results = np.array([np.vstack(target_lst), np.vstack(prediction_results_lst)])
    np.save(os.path.join(save_path, 'results.npy'), np.array(target_prediction_results))


def main():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    log_base_filepath = '/'.join(checkpoint_filepath.split('/'))
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir,
                        str(cfg.seed)])
    
    log_dir = os.path.join(log_base_filepath, log_dir)
    # log_dir = log_base_filepath

    model = ActionPredictionModel(lr=cfg.lr,
                                seed=cfg.seed,
                                if_cuda=cfg.if_cuda,
                                if_test=True,
                                gamma=cfg.gamma,
                                log_dir=log_dir,
                                train_batch=cfg.train_batch,
                                val_batch=cfg.val_batch,
                                test_batch=cfg.test_batch,
                                num_workers=cfg.num_workers,
                                data_filepath=cfg.data_filepath,
                                lr_schedule=cfg.lr_schedule,
                                time_length=cfg.time_length)

    ckpt = torch.load(checkpoint_filepath, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.freeze()
    model.eval()
    test_set_file_path = cfg.data_filepath

    test_set = PredictAction(flag='test', 
                             seed=cfg.seed, 
                             dataset_folder=test_set_file_path, 
                             time_length=cfg.time_length)


    test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                             batch_size=cfg.test_batch,
                                             shuffle=False)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed) 
    test_Transformer(test_loader, model, log_dir)
        

if __name__ == '__main__':
    main()
