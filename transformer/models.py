import os
import glob
import torch
import shutil
import numpy as np
from torch import nn
from utils import common
import pytorch_lightning as pl
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from collections import OrderedDict
from torchmetrics import classification
from torch.utils.data import DataLoader
from dataset import PredictAction
from model_utils import ActionPredictionTransformerClassificationModel

class ActionPredictionModel(pl.LightningModule):
    def __init__(self,
                 lr: float=5e-5,
                 seed: int=1,
                 if_cuda: bool=True,
                 if_test: bool=False,
                 gamma: float=0.5,
                 log_dir: str='logs',
                 train_batch: int=1400,
                 val_batch: int=1400,
                 test_batch: int=1400,
                 num_workers: int=8,
                 model_name: str='mlps',
                 data_filepath: str='data',
                 loss_type: str='ce',
                 lr_schedule: list=[100000], 
                 time_length: int=2900,
                 input_modality: list=['pupil', 'eeg', 'ekg', 'speech']) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        self.__build_model()

    def __build_model(self):
        # model
        self.model = ActionPredictionTransformerClassificationModel(in_channels=self.hparams.time_length, input_modality=self.hparams.input_modality)

        # loss
        self.loss_func = nn.CrossEntropyLoss()
        # accuracy
        # self.accuracy_func = classification.MulticlassAccuracy(3)
        # self.accuracy_func = self.correlation_arruracy()
    
    def get_tgt_mask(self, size1, size2 = 0) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        if size2 == 0:
            size2 = size1
        mask = torch.tril(torch.ones(size1, size2) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0

        return mask
    
    def correlation_arruracy(self, prediction, target):
        pred = torch.argmax(prediction, dim=2).float()
        targ = target.float()
        pearson_r_lst = []
        for i in range(len(pred)):
            pearson_r_lst.append(torch.corrcoef(torch.stack((pred[i], targ[i])))[0,1])
        return torch.nanmean(torch.tensor(pearson_r_lst))
    
    def training_step(self, batch, batch_idx):
        src, trg, trg_y = batch
        trg_mask = self.get_tgt_mask(trg.shape[1]).to(self.device)
        src_mask = self.get_tgt_mask(30, src.shape[1]-2432).to(self.device)
        pred_output = self.model(src, trg, src_mask, trg_mask) 
        train_acc = self.correlation_arruracy(pred_output, trg_y)
        train_loss = self.loss_func(pred_output.permute(0,2,1), trg_y.long())

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss



    def validation_step(self, batch, batch_idx):
        src, trg, trg_y = batch
        trg_mask = self.get_tgt_mask(trg.shape[1]).to(self.device)
        src_mask = self.get_tgt_mask(30, src.shape[1]-2432).to(self.device)
        pred_output = self.model(src, trg, src_mask, trg_mask) 
        val_acc = self.correlation_arruracy(pred_output, trg_y)
        val_loss = self.loss_func(pred_output.permute(0,2,1), trg_y.long())

        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

        
    def test_step(self, batch, batch_idx):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src, trg, trg_y = batch
        src = src.to(device)
        trg_y = trg_y.to(device)
        trg_mask = self.get_tgt_mask(trg.shape[1]).to(self.device)
        src_mask = self.get_tgt_mask(30, src.shape[1]-2432).to(self.device)
        pred_output = self.model(src, trg, src_mask, trg_mask) 
        test_acc = self.correlation_arruracy(pred_output, trg_y)
        test_loss = self.loss_func(pred_output.permute(0,2,1), trg_y.long())

        self.log('test_loss', test_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_acc', test_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return test_loss, pred_output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]

    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = PredictAction(flag='train',
                                               seed=self.hparams.seed,
                                               dataset_folder=self.hparams.data_filepath,
                                               time_length = self.hparams.time_length)
            
            self.val_dataset = PredictAction(flag='val',
                                             seed=self.hparams.seed,
                                             dataset_folder=self.hparams.data_filepath,
                                             time_length = self.hparams.time_length)
        
        if stage == 'test':
            self.test_dataset = PredictAction(flag='test',
                                              seed=self.hparams.seed,
                                              dataset_folder=self.hparams.data_filepath,
                                              time_length = self.hparams.time_length)


    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(dataset=self.train_dataset,
                                                   batch_size=self.hparams.train_batch,
                                                   shuffle=True,
                                                   **self.kwargs)
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(dataset=self.val_dataset,
                                                batch_size=self.hparams.val_batch,
                                                shuffle=False,
                                                **self.kwargs)
        return val_loader

    def test_dataloader(self):
        test_loader = torch.utils.data.DataLoader(dataset=self.test_dataset,
                                                  batch_size=self.hparams.test_batch,
                                                  shuffle=False,
                                                  **self.kwargs)
        return test_loader