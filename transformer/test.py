import os
import sys
import glob
import yaml
import copy
import torch
import pprint
import numpy as np
import pandas as pd
from utils import common
from munch import munchify
import matplotlib.pyplot as plt
from dataset import PredictAction
from scipy.spatial import distance
from scipy.signal import correlate
from collections import OrderedDict
from models import ActionPredictionModel
from torchmetrics.classification import MulticlassAUROC
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger


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

def get_prediction_results(batch_prediction, target):
    prediction_output_lst = []
    for i in range(len(batch_prediction)):
        prediction_output_lst.append(np.array(torch.argmax(batch_prediction[i], dim=1).float().to('cpu')) - 1)
    plt.figure(figsize=(200, 6), dpi=100)
    plt.plot(np.vstack(prediction_output_lst).flatten(), 'o-')
    plt.plot(target.flatten(), '.-', alpha=0.5)
    plt.savefig('prediction_results.png', bbox_inches='tight', pad_inches=0)
    plt.close()

    return np.vstack(prediction_output_lst)


def main():
    config_filepath = str(sys.argv[1])
    checkpoint_filepath = str(sys.argv[2])
    checkpoint_filepath = glob.glob(os.path.join(checkpoint_filepath, '*.ckpt'))[0]
    result_save_path = '/'.join(checkpoint_filepath.split('/')[:-2])

    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)
    
    model_name = cfg.data_filepath.split('/')[-1]

    log_dir = '_'.join([cfg.log_dir,
                        model_name,
                        'seed',
                        str(cfg.seed)])
    
    model = ActionPredictionModel(lr=cfg.lr,
                                  seed=cfg.seed,
                                  if_cuda=cfg.if_cuda,
                                  if_test=False,
                                  gamma=cfg.gamma,
                                  log_dir=cfg.log_dir,
                                  train_batch=cfg.train_batch,
                                  val_batch=cfg.val_batch,
                                  test_batch=cfg.test_batch,
                                  num_workers=cfg.num_workers,
                                  data_filepath=cfg.data_filepath,
                                  lr_schedule=cfg.lr_schedule)

    ckpt = torch.load(checkpoint_filepath)

    model.load_state_dict(ckpt['state_dict'], strict=False)
    model = model.to('cuda')

    model.eval()
    model.freeze()

    trainer = Trainer(enable_checkpointing=False, 
                      logger=False,
                      accelerator='gpu', 
                      devices=1)
    trainer.test(model)
    test_loader = model.test_dataloader()
    predictions = trainer.predict(model, test_loader)
    target = test_loader.dataset.current_data[-1]

    predicted_output = get_prediction_results(predictions,target)
    np.save(f"{result_save_path}/pred_target.npy", np.vstack([predicted_output, target]))



if __name__ == '__main__':
    main()
