import os
import sys
import yaml
import torch
import pprint
from utils import common
from munch import munchify
from models import ActionPredictionModel
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# python train.py configs/config1.yaml

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
        torch.cuda.manual_seed_all(cfg.seed)

def main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    log_dir = '_'.join([cfg.log_dir])
    
    model = ActionPredictionModel(lr=cfg.lr,
                                  seed=cfg.seed,
                                  if_cuda=cfg.if_cuda,
                                  if_test=False,
                                  gamma=cfg.gamma,
                                  log_dir=log_dir,
                                  train_batch=cfg.train_batch,
                                  val_batch=cfg.val_batch,
                                  test_batch=cfg.test_batch,
                                  num_workers=cfg.num_workers,
                                  data_filepath=cfg.data_filepath,
                                  lr_schedule=cfg.lr_schedule)


    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(filename=log_dir + "{epoch}_{val_acc:.9f}_{val_loss:.9f}",
                                          verbose=True,
                                          monitor='val_loss',
                                          mode='min')
    
    
    # define trainer
    trainer = Trainer(max_epochs=cfg.epochs,
                      default_root_dir=log_dir,
                      deterministic=True,
                      callbacks=checkpoint_callback,
                      amp_backend='native',
                      accelerator="gpu",
                      devices=1,
                      log_every_n_steps=10)

    trainer.fit(model)


if __name__ == '__main__':
    main()