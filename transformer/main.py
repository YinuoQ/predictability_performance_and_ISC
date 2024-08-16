import os
import sys
import yaml
import torch
import pprint
from munch import munchify
from models import ActionPredictionModel
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

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


def main():
    config_filepath = str(sys.argv[1])
    cfg = load_config(filepath=config_filepath)
    pprint.pprint(cfg)
    cfg = munchify(cfg)
    seed(cfg)
    seed_everything(cfg.seed)

    model_name = cfg.model_name \
                 + str(sys.argv[2]) \
                 + '-' + str(sys.argv[3]) \
                 + '-' + str(sys.argv[4]) # argv[2]: window size, argv[3]: step size, argv[4]: step number

    log_dir = '_'.join([cfg.log_dir,
                        model_name,
                        str(cfg.seed)])

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
                                  model_name=model_name,
                                  data_filepath=cfg.data_filepath,
                                  loss_type = cfg.loss_type,
                                  lr_schedule=cfg.lr_schedule,
                                  time_length=cfg.time_length,
                                  time_out=cfg.time_out)

    # define callback for selecting checkpoints during training
    checkpoint_callback = ModelCheckpoint(
        filename=log_dir + "{epoch}_{val_auroc}_{val_acc}",
        verbose=True,
        monitor='val_auroc',
        mode='max',
        prefix='')

    # define trainer
    trainer = Trainer(gpus=cfg.num_gpus,
                      max_epochs=cfg.epochs,
                      deterministic=True,
                      plugins=DDPPlugin(find_unused_parameters=False),
                      amp_backend='native',
                      default_root_dir=log_dir,
                      checkpoint_callback=checkpoint_callback)

    trainer.fit(model)


if __name__ == '__main__':
    main()