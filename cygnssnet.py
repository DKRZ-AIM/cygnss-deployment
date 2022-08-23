#! /usr/bin/env python3

import os
try:
    import nni
except ImportError:
    pass
import copy
import time
import h5py
import argparse
import numpy as np
import math
import datetime
from collections import defaultdict
import pprint

import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.callbacks.model_summary import ModelSummary

from typing import Optional

import sys
sys.path.append('../../')
import gfz_202003.utils.mathematics as mat
import gfz_202003.utils.cygnss_utils as cutils


class CyGNSSDataModule(pl.LightningDataModule):
    """ Lightning data module for CyGNSS data """

    def __init__(self, args):
        super().__init__()
        self.args = args

    def prepare_data(self, args):
        '''Use this method to do things that might write to disk or that need to be done 
        only from a single process in distributed settings.'''
        pass

    def setup(self, stage: Optional[str] = None):
        '''
        Data operations performed on every GPU

        setup() expects an stage: Optional[str] argument. It is used to separate setup logic 
        for trainer.{fit,validate,test}. If setup is called with stage = None, we assume all 
        stages have been set-up.

        Creates self.{train_data, valid_data, test_data} depending on 'stage' (CyGNSSDataset)
        '''

        if stage in (None, 'fit'):
            self.train_data = CyGNSSDataset('train', self.args)
            self.valid_data = CyGNSSDataset('valid', self.args)
        if stage in (None, 'test'):
            self.test_data = CyGNSSDataset('test', self.args)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.args.batch_size, 
                          num_workers=self.args.num_workers, shuffle=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.args.test_batch_size, 
                          num_workers=self.args.num_workers, shuffle=False, drop_last=False)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.args.test_batch_size, 
                          num_workers=self.args.num_workers, shuffle=False, drop_last=False)

    def predict_dataloader(self, args): # predicts on test set
        return DataLoader(self.test_data, batch_size=self.args.test_batch_size, 
                          num_workers=self.args.num_workers, shuffle=False, drop_last=False)

    def get_input_shapes(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            shape_data = self.train_data
        if stage == 'test':
            shape_data = self.test_data 

        if isinstance(shape_data[0][0], tuple):
            input_shapes = tuple(input_.shape for input_ in shape_data[0][0])
        else:
            input_shapes = shape_data[0][0].shape,
        return input_shapes

    @staticmethod
    def add_dataloader_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--log-transform-rain', action='store_true')
        parser.add_argument('--transform-target', action='store_true', 
                            help='Transform labels to normal distribution (experimental)')
        parser.add_argument('--confidence-interval-bottom', type=str, default='',
                            help='remove all samples below this threshold')
        parser.add_argument('--confidence-interval-top', type=str, default='',
                            help='remove all samples above this threshold')
        parser.add_argument('--min-wind-speed', type=float, default=0.0)
        parser.add_argument('--train-days', type=float, default=np.inf)
        parser.add_argument('--min-samples-in-sequence', type=int, default=1, 
                            help='Minimum samples for sequence to be considered in averaged dataset.')
        parser.add_argument('--num-workers', type=int, default=1, help='dataloader processes')
        parser.add_argument('--accept-reject', type=float, nargs=2, 
                            help='alpha+window for gfz_202003.utils.cygnss_utils.accept_reject') 
        return parser


    def print_all_quantile_losses(self, y_pred, dataloader):
        def print_quantile_losses(name, losses, quantiles):
            quantile_losses_str = ', '.join([f'{l:.2f}' for l in losses])
            quantiles_str = ', '.join([f'{q:.2f}' for q in quantiles])
            print(f'\nlosses by {name} quantile: {quantile_losses_str}\nwith quantiles: {quantiles_str}')

        def print_quantile_losses_2d(name1, name2, losses, quantiles1, quantiles2, sizes):
            print(f'\n2d quantile losses for {name1}+{name2} (rows are per quantile of {name1}):')
            for loss_row in losses:
                print(', '.join([f'{l:.2f}' for l in loss_row]))
            print('\nwith sizes:')
            for size_row in sizes:
                print(', '.join([f'{s}' for s in size_row]))
            print()
            quantiles_strs = [', '.join([f'{q:.2f}' for q in quantiles]) for quantiles in [quantiles1, quantiles2]]
            print(f'with {name1} quantiles: {quantiles_strs[0]}')
            print(f'with {name2} quantiles: {quantiles_strs[1]}')
            print()

        y_true = dataloader.dataset.y

        print(f'Note: quantile losses always report MSE. Loss_fn was {self.args.loss}\n')

        quantile_losses, quantiles = mat.losses_by_quantiles(y_true, y_true, y_pred, self.args.n_quantiles)
        print_quantile_losses('y_true', quantile_losses, quantiles)

        for v_par in self.args.v_par_eval:
            col_idx = self.args.v_par_eval.index(v_par)
            arr = dataloader.dataset.v_par_eval[:, col_idx]
            quantile_losses, quantiles = mat.losses_by_quantiles(arr, y_true, y_pred,
                                                                 self.args.n_quantiles)
            print_quantile_losses(v_par, quantile_losses, quantiles)

        for v_par_2d in self.args.v_par_eval_2d:
            names = v_par_2d.split('+')
            col_idxs = [self.args.v_par_eval.index(name) for name in names]
            arrs = [dataloader.dataset.v_par_eval[:, col_idx] for col_idx in col_idxs]
            if 'GPM_precipitation' in names:
                # filter samples with rain > 0 if given
                mask = arrs[names.index('GPM_precipitation')] > 0.0
                arr1, arr2 = [arr[mask] for arr in arrs]
                y_true, y_pred = y_true[mask], y_pred[mask]
            quantile_losses, quantiles1, quantiles2, sizes = mat.losses_by_quantiles_2d(arr1, arr2, y_true, y_pred, 10)
            print_quantile_losses_2d(names[0], names[1], quantile_losses, quantiles1, quantiles2, sizes)

class CyGNSSDataset(Dataset):
    """ Handles everything all Datasets of the different Model have in common like loading the same data files."""
    def __init__(self, flag, args):
        '''
        Load data and apply transforms during setup

        Parameters:
        -----------
        flag : string
            Any of train / valid / test. Defines dataset.
        args : argparse Namespace
            arguments passed to the main script
        -----------
        Returns: dataset
        '''
        self.args=args
        self.h5_file = h5py.File(os.path.join(args.data, flag + '_data.h5'), 'r', rdcc_nbytes=0)  # disable cache
        # load everything into memory
        print(f'\ntimestamp of {flag} data: {self.h5_file.attrs["timestamp"]}')
        start_time = time.time()
        print(f'loading {flag} data into memory...', end='\r')
        
        self.v_par_all_names = self.args.v_par + self.args.v_par_eval

        # load labels
        self.y = self.h5_file['windspeed'][:].astype(np.float32)

        # normalize main input data
        # Save normalization values together with the trained model
        # For inference load the normalization values

        if flag=='train': # determine normalization values
            self.args.normalization_values = dict()
        
        print(f'\nnormalizing {flag} data')
        # stack map vars (2D vars)
        self.X = []
        for v_map in self.args.v_map:
            X_v_map = self.h5_file[v_map][:].astype(np.float32)
            if flag=='train':
                norm_vals = dict()
                X_v_map, X_mean, X_std = mat.standard_scale(X_v_map, return_params=True)
                self.args.normalization_values[f'{v_map}_mean'] = X_mean
                self.args.normalization_values[f'{v_map}_std']  = X_std
            else:
                X_mean = self.args.normalization_values[f'{v_map}_mean']
                X_std = self.args.normalization_values[f'{v_map}_std']
                X_v_map = mat.standard_scale(X_v_map, mean=X_mean, scale=X_std)
                
            self.X.append(X_v_map) # append scaled 2D map
            print(f'* {v_map} after normalization: mean={np.mean(X_v_map):.2e} / std = {np.std(X_v_map):.2e}')
        self.X = np.stack(self.X, axis=1)

        # stack additional vars
        # load all vars first
        self.v_par_all = [self.h5_file[par][:] for par in self.v_par_all_names]
        n_v_par = len(self.args.v_par)
        if n_v_par > 0:
            # only normalize input vars
            for i, par in enumerate(self.v_par_all[:n_v_par]):
                key = self.v_par_all_names[i]
                # additional log transform
                if key in self.args.log_v_par:
                    print(f'* {key} is log-transformed before normalization')
                    par = mat.log_transform(par)
                if flag=='train':
                    self.v_par_all[i], v_mean, v_std = mat.standard_scale(par, return_params=True) 
                    self.args.normalization_values[f'{key}_mean'] = v_mean
                    self.args.normalization_values[f'{key}_std'] = v_std
                else:
                    v_mean = self.args.normalization_values[f'{key}_mean']
                    v_std = self.args.normalization_values[f'{key}_std']
                    self.v_par_all[i] = mat.standard_scale(par, mean=v_mean, scale=v_std)
                tm = np.mean(self.v_par_all[i]) # true mean
                ts = np.std(self.v_par_all[i]) # true std
                print(f'* {key} after normalization: mean={tm:.2e} / std = {ts:.2e}')
            self.v_par = np.stack(self.v_par_all[:n_v_par], axis=1)
        else:
            self.v_par = []
        if self.args.v_par_eval:
            self.v_par_eval = np.stack(self.v_par_all[n_v_par:], axis=1)
        else:
            self.v_par_eval = []

        # additional reduction of the data set
        if self.args.min_samples_in_sequence > 1:
            col_idx = self.args.v_par_eval.index('samples_in_sequence')
            mask = self.v_par_eval[:, col_idx] >= self.args.min_samples_in_sequence
            mask = mask.astype(bool)
            self._filter_all_data_by_mask(mask, flag, 'min_samples_in_sequence')
        if self.args.min_wind_speed > 0:
            col_idx = self.args.v_par_eval.index('windspeed')
            mask = self.v_par_eval[:, col_idx] >= self.args.min_wind_speed
            mask = mask.astype(bool)
            self._filter_all_data_by_mask(mask, flag, 'min_wind_speed')
        if flag=='train' and self.args.train_days < np.inf:
            col_idx = self.args.v_par_eval.index('ddm_timestamp_unix')
            daystamp = self.v_par_eval[:, col_idx]
            daystamp = (daystamp - np.min(daystamp)) / 24 / 3600
            # select all samples with days within (dmax - train_days)
            mask = daystamp >= np.max(daystamp) - self.args.train_days
            print(daystamp[mask])
            self._filter_all_data_by_mask(mask, flag, 'train_days')
        if self.args.confidence_interval_bottom:
            col_idx = self.args.v_par_eval.index(self.args.confidence_interval_bottom)
            col_idx = self.args.v_par_eval.index(self.args.confidence_interval_bottom)
            mask = self.v_par_eval[:, col_idx].astype(bool)
            self._filter_all_data_by_mask(mask, flag, \
                'confidence interval (bottom) ' + self.args.confidence_interval_bottom)
        if self.args.confidence_interval_top:
            col_idx = self.args.v_par_eval.index(self.args.confidence_interval_top)
            col_idx = self.args.v_par_eval.index(self.args.confidence_interval_top)
            mask = self.v_par_eval[:, col_idx].astype(bool)
            mask = ~mask # *exclude* the samples exceeding the top confidence interval
            self._filter_all_data_by_mask(mask, flag, \
                'confidence interval (top) ' + self.args.confidence_interval_top)
        if self.args.accept_reject:
            if flag=='train':
                alpha = self.args.accept_reject[0]
                window = int(self.args.accept_reject[1])
                mask = cutils.accept_reject(self.y, alpha, window)
                self._filter_all_data_by_mask(mask, flag, 'accept reject')
        if self.args.transform_target: # careful to transform only after cutting confidence intervals!
            print('Power transform with scale = ', self.args.transform_target_scale)
            y_tmp = mat.power_transform(self.y, self.args.transform_target_scale)
            self.y = y_tmp

            self.y -= self.args.transform_target_mean
            self.y /= self.args.transform_target_std
            print('Transformed target:')
            print('Mean', np.mean(self.y))
            print('Std', np.std(self.y))

        print(f'loading and transforming all {flag} data in memory took {time.time() - start_time:.2f}s')
        print(f'{flag} input data: {self.X.shape} ({self.X.nbytes // 1e6}MB)')
        print(f'{flag} labels: {self.y.shape} ({self.y.nbytes // 1e6}MB)')

    def _filter_all_data_by_mask(self, mask, flag, name=''):
        self.X, self.y = self.X[mask], self.y[mask]
        if len(self.v_par) > 0:
            self.v_par = self.v_par[mask]
        if len(self.v_par_eval) > 0:
            self.v_par_eval = self.v_par_eval[mask]
        print(f'{flag} input data after {name} downsampling: {self.X.shape} ({self.X.nbytes // 1e6}MB)')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx]
        if len(self.v_par) > 0:
            X = (X, self.v_par[idx])
        y = self.y[idx]
#        if self.args.auxiliary_loss_rain:
#            y_aux = self.v_par_eval[idx, self.args.v_par_eval.index('GPM_precipitation')]
#            y = (y, y_aux)
        return (X, y)

class ImageNet(pl.LightningModule):
    def __init__(self, args, input_shapes):
        super().__init__()
        # model set up goes here
        self.args = args
        n_channels, L1, L2 = input_shapes[0]
        if args.v_par:
            L3 = input_shapes[1][0]
        n_output_values = 1
#        n_output_values = 2 if args.auxiliary_loss_rain else 1

        final_conv_filters = args.filters_conv1 * (2 ** ((args.n_conv_layers - 1)// args.double_filters_every))
        pool = 2
        n_poolings_max = int(math.log(min(L1, L2) - 3, pool))  # leave at least a 3x3 square
        n_poolings = min(args.n_conv_layers // args.pool_every, n_poolings_max)
        S = final_conv_filters * (L1 // (pool ** n_poolings)) * (L2 // (pool ** n_poolings))
        if args.v_par:
            S += args.units_dense_v_par

        self.activation = CyGNSSNet.activation_fn(args.activation)
        self.activation_v_par = CyGNSSNet.activation_fn(args.activation_v_par)
        self.loss = CyGNSSNet.loss_fn(args.loss) 

        self.cv1 = torch.nn.Conv2d(n_channels, args.filters_conv1, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(args.filters_conv1)

        self.hidden = torch.nn.ModuleList()
        for i in range(args.n_conv_layers - 1):
            filters_conv_in = args.filters_conv1 * (2 ** (i // args.double_filters_every))
            filters_conv_out = args.filters_conv1 * (2 ** ((i + 1) // args.double_filters_every))
            if (i + 1) % args.pool_every == 0 and (i + 1) // args.pool_every < n_poolings_max + 1:
                self.hidden.append(torch.nn.MaxPool2d(pool))
            self.hidden.append(torch.nn.Conv2d(filters_conv_in, filters_conv_out, kernel_size=3, padding=1))
            self.hidden.append(torch.nn.BatchNorm2d(filters_conv_out))
        # in some cases it may be necessary to add pooling here as well
        if (args.n_conv_layers % args.pool_every == 0
            and args.n_conv_layers // args.pool_every <= n_poolings_max):
            self.hidden.append(torch.nn.MaxPool2d(pool))

        if args.v_par:
            self.fc_v_par = torch.nn.Linear(L3, args.units_dense_v_par)
            self.dr_v_par = torch.nn.Dropout(args.dropout_v_par)

        self.fc1 = torch.nn.Linear(S, args.units_dense1)
        self.dr_fc1 = torch.nn.Dropout(args.dropout_dense1)
        if args.units_dense2 > 0:
            self.fc2 = torch.nn.Linear(args.units_dense1, args.units_dense2)
            self.dr_fc2 = torch.nn.Dropout(args.dropout_dense2)
            self.fc_final = torch.nn.Linear(args.units_dense2, n_output_values)
        else:
            self.fc_final = torch.nn.Linear(args.units_dense1, n_output_values)

    def forward(self, x):
        # forward pass goes here
        if self.args.v_par:
            x, x_v_par = x
        x = x.float()
        x = self.activation(self.cv1(x))
        x = self.bn1(x)

        for i in range(len(self.hidden)):
            if ((self.args.bn_after_activation and isinstance(self.hidden[i], torch.nn.Conv2d)) or
                (not self.args.bn_after_activation and isinstance(self.hidden[i], torch.nn.BatchNorm2d))):
                x = self.activation(x)
            x = self.hidden[i](x)

        x = torch.flatten(x, 1)

        if self.args.v_par:
            x_v_par = x_v_par.float()
            x_v_par = self.activation_v_par(self.fc_v_par(x_v_par))
            x_v_par = self.dr_v_par(x_v_par)
            x = torch.cat((x, x_v_par), 1)

        x = self.activation(self.fc1(x))
        x = self.dr_fc1(x)
        if self.args.units_dense2 > 0:
            x = self.activation(self.fc2(x))
            x = self.dr_fc2(x)
        x = self.fc_final(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch-size', type=int, default=256)
        parser.add_argument('--filters-conv1', type=int, default=32)
        parser.add_argument('--n-conv-layers', type=int, default=4)
        parser.add_argument('--double-filters-every', type=int, default=2)
        parser.add_argument('--pool-every', type=int, default=2,
                                 help='To disable pooling make this greater than --n-conv-layers')
        parser.add_argument('--units-dense1', type=int, default=64)
        parser.add_argument('--units-dense2', type=int, default=64)
        parser.add_argument('--units-dense-v-par', type=int, default=32)
        parser.add_argument('--activation', type=str, default='relu')
        parser.add_argument('--activation-v-par', type=str, default='relu')
        parser.add_argument('--loss', type=str, choices=['mae', 'mse'], default='mse')
        parser.add_argument('--bn-after-activation', action='store_true')
        parser.add_argument('--dropout-dense1', type=float, default=0.0)
        parser.add_argument('--dropout-dense2', type=float, default=0.0)
        parser.add_argument('--dropout-v-par', type=float, default=0.0)
        return parser

class DenseNet(pl.LightningModule):
    def __init__(self, args, input_shapes):
        super().__init__()
        # model set up goes here
        self.args = args
        assert self.args.v_par, 'need --v-par values to predict on'
        self.activation = CyGNSSNet.activation_fn(self.args.activation)
        self.loss = CyGNSSNet.loss_fn(args.loss)

        n_channels, L1, L2 = input_shapes[0]
        if args.v_par:
            L3 = input_shapes[1][0]
        n_output_values = 1

        n_input_values = n_channels * L1 * L2 + L3

        self.fc1 = torch.nn.Linear(n_input_values, self.args.units_dense1)
        self.dr_fc1 = torch.nn.Dropout(self.args.dropout_dense1)
        self.fc2 = torch.nn.Linear(self.args.units_dense1, self.args.units_dense2)
        self.dr_fc2 = torch.nn.Dropout(self.args.dropout_dense2)
        self.fc_final = torch.nn.Linear(self.args.units_dense2, n_output_values)

    def forward(self, x):
        x_map, x_v_par = x
        x_map = torch.flatten(x_map, 1)
        x = torch.cat((x_map, x_v_par), 1)
        x = self.dr_fc1(self.activation(self.fc1(x)))
        x = self.dr_fc2(self.activation(self.fc2(x)))
        x = self.fc_final(x)
        return x

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--batch-size', type=int, default=256)
        parser.add_argument('--activation', type=str, default='relu')
        parser.add_argument('--units-dense1', type=int, default=64)
        parser.add_argument('--units-dense2', type=int, default=64)
        parser.add_argument('--dropout-dense1', type=float, default=0.0)
        parser.add_argument('--dropout-dense2', type=float, default=0.0)
        return parser

class CyGNSSNet(pl.LightningModule):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        #self.save_hyperparameters(self.backbone.args)
        self.best_loss = np.inf # reported for nni
        self.best_epoch = 0

    def forward(self, x):
        y = self.backbone(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x)
        y_pred = torch.squeeze(y_pred, dim=1)
        loss = self.backbone.loss(y_pred, y)
#        self.log('train_loss', loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x)
        y_pred = torch.squeeze(y_pred, dim=1)
        loss = self.backbone.loss(y_pred, y)
        self.log('valid_loss', loss, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs):
        val_loss = self.trainer.callback_metrics["valid_loss"]
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.best_epoch = self.trainer.current_epoch
        self.log('best_loss', self.best_loss)
        self.log('best_epoch', self.best_epoch)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x)
        y_pred = torch.squeeze(y_pred, dim=1)
        loss = self.backbone.loss(y_pred, y)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.backbone(x)
        y_pred = torch.squeeze(y_pred, dim=1)
        return y_pred

    def configure_optimizers(self):
        if self.backbone.args.optimizer=='adam':
            return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        elif self.backbone.args.optimizer=='sgd':
            return torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate)

    def configure_callbacks(self):
        '''Model checkpoint callback goes here'''
        callbacks = [ModelCheckpoint(monitor='valid_loss', mode='min',
                     dirpath=os.path.join(os.path.dirname(self.backbone.args.save_model_path), 'checkpoint'),
                     filename="cygnssnet-{epoch}")]
        return callbacks

    @staticmethod
    def activation_fn(activation_name):
        if activation_name == 'tanh':
            return torch.tanh
        elif activation_name == 'relu':
            return F.relu
        elif activation_name == 'sigmoid':
            return torch.sigmoid
        elif activation_name == 'leaky_relu':
            return F.leaky_relu

    @staticmethod
    def loss_fn(loss_name):
        if loss_name == 'mse':
            return F.mse_loss
        elif loss_name == 'mae':
            return F.l1_loss

    @staticmethod
    def best_checkpoint_path(save_model_path, best_epoch):
        '''Path to best checkpoint'''
        ckpt_path = os.path.join(os.path.dirname(save_model_path), 'checkpoint', f"cygnssnet-epoch={best_epoch}.ckpt")
        all_ckpts = os.listdir(os.path.dirname(ckpt_path))
        # If the checkpoint already exists, lightning creates "*-v1.ckpt"
        only_ckpt = ~np.any([f'-v{best_epoch}' in ckpt for ckpt in all_ckpts])
        assert only_ckpt, f'Cannot load checkpoint: found versioned checkpoints for best_epoch {best_epoch} in {os.path.dirname(ckpt_path)}'
        return ckpt_path

class CyGNSSMetricCallbacks(Callback):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def on_validation_epoch_end(self, trainer, pl_module):
        '''After each epoch metrics on validation set'''
        if self.args.nni:
            nni.report_intermediate_result(float(trainer.callback_metrics['valid_loss']))
        metrics = trainer.callback_metrics # everything that was logged in self.log
        epoch = trainer.current_epoch
        print(f'Epoch {epoch} metrics:')
        for key, item in metrics.items():
            print(f'  {key}: {item:.4f}')

    def on_train_epoch_start(self, trainer, pl_module):
        print(f'\nEpoch {trainer.current_epoch} starts training ...')
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        tt = time.time() - self.epoch_start_time
        print(f'Epoch {trainer.current_epoch} finished training in {tt:.0f} seconds')

    def on_test_epoch_end(self, trainer, pl_module):
        pass

    def on_epoch_end(self, trainer, pl_module):
        '''After each epoch (T+V)'''
        pass

    def on_train_end(self, trainer, pl_module):
        '''Final metrics on validation set (after training is done)'''
        print(f'Finished training in {trainer.current_epoch+1} epochs')
        if self.args.nni:
            nni.report_final_result(float(trainer.callback_metrics['best_loss']))

    @staticmethod
    def add_nni_params(args):
        args_nni = nni.get_next_parameter()
        assert all([key in args for key in args_nni.keys()]), 'need only valid parameters'
        args_dict = vars(args)
        # cast params that should be int to int if needed (nni may offer them as float)
        args_nni_casted = {key:(int(value) if type(args_dict[key]) is int else value)
                            for key,value in args_nni.items()}
        args_dict.update(args_nni_casted)

        # adjust paths to NNI_OUTPUT_DIR (overrides passed args)
        nni_output_dir = os.path.expandvars('$NNI_OUTPUT_DIR')
        for param in ['save_model_path', 'prediction_output_path']:
            nni_path = os.path.join(nni_output_dir, os.path.basename(args_dict[param]))
            args_dict[param] = nni_path
        return args

def main():
    # ----------
    # args
    # ----------
    
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--model', type=str, choices=['cnn', 'dense'], default='cnn',
                         help='''Model architecture. 
                                 cnn - ImageNet, process 2D maps in convolutional layers,
                                 dense - DenseNet, flatten all inputs''')
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--test-batch-size', type=int, default=512, help='Larger batch size for validation data')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    # data
    parser.add_argument('--data', type=str, help='should enlist train_data.h5, valid_data.h5, (test_data.h5)')
    parser.add_argument('--v-map', type=str, nargs='+', choices=['brcs', 'eff_scatter', 'raw_counts', 'power_analog'],
                             default=['brcs'], help='Data column to use as conv input')
    parser.add_argument('--v-par', type=str, nargs='*', default=[],
                             help='Continuous data columns to use as addtional inputs')
    parser.add_argument('--v-par-eval', type=str, nargs='*',
                             default=['GPM_precipitation', 'ddm_ant', 'ddm_brcs_uncert',
                                      'ddm_les', 'ddm_nbrcs', 'ddm_snr', 'ddm_timestamp_unix', 'ddm_timestamp_utc',
                                      'gps_eirp', 'les_scatter_area_log10', 'lna_temp_nadir_port',
                                      'lna_temp_nadir_starboard', 'lna_temp_zenith', 'nbrcs_scatter_area_log10',
                                      'rx_to_sp_range_log10', 'samples_in_sequence',
                                      'sc_alt', 'sc_lat', 'sc_lon', 'sc_pitch', 'sc_roll', 'sc_velocity', 'sc_yaw',
                                      'sp_alt', 'sp_az_orbit', 'sp_inc_angle', 'sp_lat', 'sp_lon', 'sp_rx_gain',
                                      'sp_theta_orbit', 'sp_velocity', 'sv_num', 'track_id',
                                      'tx_to_sp_range_log10', 'windspeed', 'zenith_code_phase', 'zenith_sig_i2q2',
                                      'ddm_nbrcs_exceeds_alpha_0.975', 'ddm_nbrcs_exceeds_alpha_0.025'],
                             help='Continuous data columns to evaluate by')
    parser.add_argument('--v-par-eval-2d', type=str, nargs='*', default=['windspeed+GPM_precipitation'],
                             help='Additional variables to be evaluated together, seperated by a "+" sign')
    parser.add_argument('--log-v-par', type=str, nargs='*', default=[], help='Input parameters to be log-transformed')
    # callbacks & logs
    parser.add_argument('--n-quantiles', type=int, default=20, help='for quantile loss evaluation')
    parser.add_argument('--early-stopping', dest='early_stopping', action='store_true')
    parser.add_argument('--no-early-stopping', dest='early_stopping', action='store_false')
    parser.set_defaults(early_stopping=True)
    parser.add_argument('--patience', type=int, default=3, 
                         help='Epochs to wait before early stopping')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--nni', action='store_true')
    # store and load
    parser.add_argument('--save-model-path', type=str, default='./best_model.pt')
    parser.add_argument('--prediction-output-path', type=str, default='best_predictions.h5')
    parser.add_argument('--load-model-path', type=str, default='')

    parser = pl.Trainer.add_argparse_args(parser)
    parser = CyGNSSDataModule.add_dataloader_specific_args(parser)

    # add model specific args depending on chosen model
    temp_args, _ = parser.parse_known_args()
    if temp_args.model=='cnn':
        parser = ImageNet.add_model_specific_args(parser)
    elif temp_args.model=='dense':
        parser = DenseNet.add_model_specific_args(parser)

    args = parser.parse_args()

    if args.nni:
        args = CyGNSSMetricCallbacks.add_nni_params(args)

    if args.verbose:
        print('BEGIN argparse key - value pairs')
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(args))
        print('END argparse key - value pairs')

    if args.load_model_path:
        print('INFERENCE MODE')
        print(f'loading model from {args.load_model_path}')

        # ----------
        # data
        # ----------

        # load arg Namespace from checkpoint
        print('command line arguments will be replaced with checkpoint["hyper_parameters"]')
        checkpoint = torch.load(args.load_model_path)
        checkpoint_args = argparse.Namespace(**checkpoint["hyper_parameters"])

        # potentially overwrite the data arg
        if args.data:
            checkpoint_args.data = args.data
            print(f'overwriting checkpoint argument: data dir = {checkpoint_args.data}')

        # potentially overwrite the vmin argument
        if args.min_wind_speed:
            checkpoint_args.min_wind_speed = args.min_wind_speed
            print(f'overwriting checkpoint argument: min_wind_speed = {checkpoint_args.min_wind_speed}')

        cdm = CyGNSSDataModule(checkpoint_args)
        cdm.setup(stage='test')
        test_loader = cdm.test_dataloader()
        input_shapes = cdm.get_input_shapes(stage='test')
        
        if args.verbose:
            print('Input shapes', input_shapes)

        if checkpoint_args.model=='cnn':
            backbone = ImageNet(checkpoint_args, input_shapes)
        elif checkpoint_args.model=='dense':
            backbone = DenseNet(checkpoint_args, input_shapes)
        # load model state from checkpoint
        model = CyGNSSNet(backbone)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        trainer = pl.Trainer(weights_summary='full', 
                             num_sanity_val_steps=0, 
                             gpus=int(torch.cuda.is_available()),
                             enable_progress_bar=False)
        trainer.test(model=model, test_dataloaders=test_loader)
        y_pred = trainer.predict(model=model, dataloaders=[test_loader])
        y_pred = torch.cat(y_pred).detach().cpu().numpy().squeeze()
        
        cdm.print_all_quantile_losses(y_pred, test_loader)
    else:
        print('TRAINING MODE')

        # ----------
        # data
        # ----------

        cdm = CyGNSSDataModule(args)
        cdm.setup(stage='fit')
        train_loader = cdm.train_dataloader()
        valid_loader = cdm.val_dataloader()

        input_shapes = cdm.get_input_shapes() 
    
        if args.verbose:
            print('Input shapes', input_shapes)
        # ----------
        # model
        # ----------
        if args.model=='cnn':
            model = CyGNSSNet(ImageNet(args, input_shapes))
        elif args.model=='dense':
            model = CyGNSSNet(DenseNet(args, input_shapes))

        # ----------
        # training
        # ----------
        callbacks = [CyGNSSMetricCallbacks(args), ModelSummary(max_depth=-1)] # model checkpoint is a model callback
        if args.early_stopping:
            callbacks.append(EarlyStopping(monitor='valid_loss', patience=args.patience, mode='min'))

        trainer = pl.Trainer.from_argparse_args(args, 
                                                fast_dev_run=False, # debug option
                                                logger=False,
                                                callbacks=callbacks, 
                                                enable_progress_bar=False,
                                                num_sanity_val_steps=0) # skip validation check
        trainer.fit(model, train_loader, valid_loader)

        best_epoch = int(trainer.callback_metrics["best_epoch"])
        ckpt_path = CyGNSSNet.best_checkpoint_path(args.save_model_path, best_epoch)
        print(f'\nLoading best model from {ckpt_path}')
        trainer.validate(dataloaders=valid_loader, ckpt_path=ckpt_path)
        #model.load_state_dict(torch.load(ckpt_path)['state_dict'])
        model.eval()
        # make predictions on *validation set*
        y_pred = trainer.predict(model=model, dataloaders=[valid_loader])
        y_pred = torch.cat(y_pred).detach().cpu().numpy().squeeze()
        # save model -- TODO breaks
        #script = model.to_torchscript()
        #torch.jit.save(script, args.save_model_path)

        cdm.print_all_quantile_losses(y_pred, valid_loader)

    # procedures that take place in fit and in test stage
    # save predictions
    h5_file = h5py.File(args.prediction_output_path, 'w')
    chunk_size = 100000 if y_pred.shape[0] >= 100000 else y_pred.shape[0]
    dset = h5_file.create_dataset('/y_pred', 
                                  shape=y_pred.shape, 
                                  chunks=(chunk_size,) + y_pred.shape[1:], 
                                  fletcher32=True, 
                                  dtype='float32')
    dset[:] = y_pred
    h5_file.attrs['timestamp'] = str(datetime.datetime.now())
    h5_file.close()
    print(f'saved predictions to {args.prediction_output_path}')

if __name__=='__main__':
    main()
