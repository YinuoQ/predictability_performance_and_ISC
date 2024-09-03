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
from model_utils import CrossModalTransformer

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
                 data_filepath: str='data',
                 lr_schedule: list=[100000]) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.kwargs = {'num_workers': self.hparams.num_workers, 'pin_memory': True} if self.hparams.if_cuda else {}
        self.__build_model()

    def __build_model(self):
        # model
        self.model = CrossModalTransformer()
        # loss
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        class_weights = torch.tensor([2.9, 0.3, 1.7]).to(device)
        self.loss_func = nn.CrossEntropyLoss(reduction='none', weight=class_weights)

        # accuracy
        self.accuracy_func = classification.Accuracy(task="multiclass", num_classes=3)


    def correlation_arruracy(self, prediction, target):
        pred = torch.argmax(prediction, dim=1).float()
        targ = target+1
        pearson_r_lst = []
        for i in range(len(pred)):
            if (pred[i] == targ[i]).sum() == 30:
                pearson_r_lst.append(torch.Tensor([1]))
            elif (pred[i] == pred[i,0]).sum() == 30 or (targ[i] == targ[i,0]).sum() == 30:
                pearson_r_lst.append(torch.Tensor([0]))
            else:
                pearson_r_lst.append(torch.abs(torch.corrcoef(torch.stack((pred[i], targ[i])))[0,1]))
        output = torch.nanmean(torch.FloatTensor(pearson_r_lst)) 
        # output = self.accuracy_func(pred, targ)
        return output
    
    def training_step(self, batch, batch_idx):
        src1, src2, src3, src4, src5, trg_y = batch
        pred_output = self.model(src1, src2, src3, src4, src5) 

        train_acc = self.correlation_arruracy(pred_output, trg_y)
        train_loss = self.loss_func(pred_output, trg_y.long()+1).mean()

        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss



    def validation_step(self, batch, batch_idx):
        src1, src2, src3, src4, src5, trg_y = batch
        pred_output = self.model(src1, src2, src3, src4, src5) 
        val_acc = self.correlation_arruracy(pred_output, trg_y)
        val_loss = self.loss_func(pred_output, trg_y.long()+1).mean()
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_acc', val_acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return val_loss

        
    def test_step(self, batch, batch_idx):
        torch.set_grad_enabled(True)

        src1, src2, src3, src4, src5, trg_y = batch   
        pred_output = self.model(src1, src2, src3, src4, src5)     
        test_acc = self.correlation_arruracy(pred_output, trg_y)
        test_loss = self.loss_func(pred_output, trg_y.long()+1).mean()

        self.log('test_loss', test_loss)
        self.log('test_acc', test_acc)
        return test_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.lr_schedule, gamma=self.hparams.gamma)
        return [optimizer], [scheduler]
    
    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        src1, src2, src3, src4, src5, trg_y = batch   
        pred_output = self.model(src1, src2, src3, src4, src5)     

        return pred_output
    
    def setup(self, stage=None):
        if stage == 'fit':
            self.train_dataset = PredictAction(flag='train',
                                               seed=self.hparams.seed,
                                               dataset_folder=self.hparams.data_filepath)
            
            self.val_dataset = PredictAction(flag='validation',
                                             seed=self.hparams.seed,
                                             dataset_folder=self.hparams.data_filepath)
        
        if stage == 'test':
            self.test_dataset = PredictAction(flag='test',
                                              seed=self.hparams.seed,
                                              dataset_folder=self.hparams.data_filepath)


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
