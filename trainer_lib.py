__author__ = 'Aaron Woodhouse'

"""
Model Trainer class.
"""

# Imports #
import time

import pandas as pd
import numpy as np
import torch
import itertools
from torch import (nn, optim)
from torchmetrics import Accuracy
from torch.utils.data import DataLoader

# Classes #
class Trainer:
    """Model trainer."""
    
    def __init__(self, model, train_dataloader, val_dataloader):
        """Initialize class.

        Parameters
        ----------
        model
            Instance of model to train.
        train_dataloader : DataLoader
            Training dataloader.
        val_dataloader : DataLoader
            Validation dataloader.

        Attributes
        ----------
        model
            Model instance to train.
        train_dataloader
            Training dataloader.
        val_dataloader
            Validation dataloader.
        optimizer
            Model optimizer.
        loss
            Loss function.

        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optim.Adam(model.parameters())
        self.loss = nn.CrossEntropyLoss()
        
    def _train_one_epoch(self, max_batches=None):
        """Train model for one epoch.

        Parameters
        ----------
        max_batches : int, optional
            Number of batches to train in epoch. The default is None.

        Returns
        -------
        float
            Average loss.
        float
            Average accuracy.

        """
        l_list = []
        acc_list = []
        for (xs, xs2, targets) in itertools.islice(self.train_dataloader, 0, max_batches):
            
            accuracy = Accuracy(task='multiclass', num_classes=7)
            pred = self.model(xs, xs2)
            
            loss = self.loss(pred, targets)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
                
            l_list.append(loss.item())
            acc_list.append(accuracy(pred, targets).item())
            
        return np.mean(l_list), np.mean(acc_list)
    
    def _val_one_epoch(self):
        """Validation of model for one epoch.

        Returns
        -------
        float
            Average loss.
        float
            Average accuracy.

        """
        l_list = []
        acc_list = []
        with torch.no_grad():
            
            accuracy = Accuracy(task='multiclass', num_classes=7)
            for (xs, xs2, targets) in self.val_dataloader:
                
                pred = self.model(xs, xs2)
                loss = self.loss(pred, targets)
            
                l_list.append(loss.item())
                acc = accuracy(pred, targets)
            
        return np.mean(l_list), np.mean(acc.item())
                
    def train(self, epochs, max_batches=None):
        """Train the model.

        Parameters
        ----------
        epochs : int
            Number of epochs to train the model for.
        max_batches : int, optional
            Number of batches to use in each epoch. The default is None.

        Returns
        -------
        :pd:Dataframe
            Training statistics.

        """
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'epoch_duration': [],
        }

        start0 = time.time()

        for epoch in range(epochs):
            start = time.time()
            
            train_loss, train_acc = self._train_one_epoch(max_batches)
            val_loss, val_acc = self._val_one_epoch()

            duration = time.time() - start
            history['train_loss'].append(train_loss)
            history['train_accuracy'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['epoch_duration'].append(duration)
            
            print("[%d (%.2fs)]: train_loss=%.2f train_acc=%.2f, val_loss=%.2f val_acc=%.2f" % (
                epoch, duration, train_loss, train_acc, val_loss, val_acc))
            
        duration0 = time.time() - start0
        print("== Total training time %.2f seconds ==" % duration0)

        return pd.DataFrame(history)
    
    def reset(self):
        """Reset model parameters."""
        
        for layer in self.model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()