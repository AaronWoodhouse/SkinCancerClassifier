#!/usr/bin/env python
__author__ = 'Aaron Woodhouse'

"""
Training and testing different models on a given dataset.
"""

# Imports #
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import nn
from torch import optim
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Dataset,
    random_split
)
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch.optim import (Optimizer, Adam)
import torchvision
from torchvision import datasets, transforms
from torchmetrics import Accuracy

DIR = os.path.dirname(os.path.realpath(__file__))
#PARENT = os.path.dirname(DIR)
sys.path.append(DIR)

import trainer_lib
import models

# Classes #
class ImageDataset(Dataset):
    """Dataset of training images."""
    
    def __init__(self, data: str, imgs: str):
        """Initialize class.
        
        Parameters
        ----------
        data : str
            Data file path.
        imgs : str
            Images folder path.
        
        Attributes
        ----------
        _x_train_img : :float32:tensor
            Training image data.
        _x_train_data : :float32:tensor
            Training metadata.
        _y_train : :long:tensor
            Training labels.

        """
        table_tensor, label_tensor, img_tensor = self.__preprocess_data(data, imgs)
        
        self._x_train_img = img_tensor.to(torch.float32)
        self._x_train_data = table_tensor.to(torch.float32)
        self._y_train = label_tensor.to(torch.long)
    
    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.y_train)
    
    def __getitem__(self, idx):
        """Get training images, metadata, and labels at a given index."""
        return self.x_train_img[idx], self.x_train_data[idx], self.y_train[idx]
    
    @property
    def x_train_img(self):
        """tensor: set training images."""
        return self._x_train_img
    
    @x_train_img.setter
    def x_train_img(self, new_val):
        self._x_train_img = new_val
    
    @x_train_img.deleter
    def x_train_img(self):
        del self._x_train_img
        
    @property
    def x_train_data(self):
        """tensor: set training data."""
        return self._x_train_data
    
    @x_train_data.setter
    def x_train_data(self, new_val):
        self._x_train_data = new_val
    
    @x_train_data.deleter
    def x_train_data(self):
        del self._x_train_data   
        
    @property
    def y_train(self):
        """tensor: set training labels."""
        return self._y_train
    
    @y_train.setter
    def y_train(self, new_val):
        self._y_train = new_val
    
    @y_train.deleter
    def y_train(self):
        del self._y_train   
    
    def __preprocess_data(self, data: str, imgs: str):
        """Preprocess training data.
        
        Parameters
        ----------
        data : str
            Data file path.
        imgs : str
            Images folder path.
        
        Returns
        -------        
        table_tensor : tensor
            Training metadata.
        label_tensor : tensor
            Training labels.
        img_tensor : tensor
            Training image data.
        """
        
        # Table Data #
        table_data = pd.read_csv(data)
        table_data = table_data.drop(['lesion_id','dx_type'], axis=1)
        
        labels = table_data['dx']
        sex = table_data['sex']
        localization = table_data['localization']

        # Make maps of unique values #
        labels_map = ['bcc','bkl','df','nv','vasc','mel','akiec']
        sex_map = sex.unique()
        localization_map = localization.unique()

        # Make data numeric #
        for i, x in enumerate(sex_map):
            indicies = np.where(sex == x)[0]
            sex.loc[indicies] = i

        for i, x in enumerate(localization_map):
            indicies = np.where(localization == x)[0]
            localization.loc[indicies] = i

        for i, x in enumerate(labels_map):
            indicies = np.where(labels == x)[0]
            labels.loc[indicies] = i

        # Remove nan samples from data #
        nan_rows = table_data.isna().any(axis=1)
        table_data = table_data.drop(nan_rows[nan_rows].index)
        labels = labels.drop(nan_rows[nan_rows].index)
        # img data inherently removes nan
            
        # Load images as tensors #
        tensors = []
        convert_tensor = transforms.ToTensor()
        
        # Only gets images that exist in metadata
        for x in table_data['image_id']:
            img = Image.open(imgs + '/' + x + ".jpg")
            
            img_t = convert_tensor(img)
            tensors.append(img_t)
            img.close()
        
        # Combine Tensors #
        img_tensor = torch.stack((tensors[0],tensors[1]), 0)
        for t in tensors[2:]:
            t = torch.unsqueeze(t, 0)
            img_tensor = torch.cat((img_tensor,t), 0)
            
        table_data = table_data.drop(['image_id','dx'], axis=1)
        table_tensor = torch.tensor(table_data.values.astype(float))
        label_tensor = torch.tensor(labels.values.astype(int))
        
        return table_tensor, label_tensor, img_tensor

# Methods #
def get_dataloaders(data: str, imgs: str):
    """Gets dataloaders for training, testing, and validation.

    Parameters
    ----------
    data : str
        Data file path.
    imgs : str
        Images folder path.

    Returns
    -------
    train_dataloader : Dataloader
        Training Dataloader.
    val_dataloader : Dataloader
        Validation Dataloader.
    test_dataloader : Dataloader
        Testing Dataloader.

    """
    df = ImageDataset(data, imgs)
    
    train_dataset, test_dataset = random_split(df, (0.8, 0.2))
    train_dataset, val_dataset = random_split(train_dataset, (0.9, 0.1))

    train_dataloader = DataLoader(train_dataset, batch_size=30, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    test_dataset = test_dataset[:][:2]    

    return train_dataloader, val_dataloader, test_dataloader
    
def train_model(model, num_epochs: int, batches: int, train_dataloader, val_dataloader):
    """Train a given model.

    Parameters
    ----------
    model
        Model to train.
    num_epochs : int
        Number of epochs to use for training.
    batches : int
        Number of images in a training batch.
    train_dataloader : DataLoader
        Training dataloader.
    val_dataloader : DataLoader
        Validation dataloader.
        
    Returns
    -------
    model_log
        Statistics of the trained model.

    """
    trainer = trainer_lib.Trainer(model, train_dataloader, val_dataloader)
    trainer.reset()
    model_log = trainer.train(epochs=num_epochs, max_batches=batches)
    model_log.round(2)
    return model_log

def test_model(model, test_dataloader):
    """Test a given model.

    Parameters
    ----------
    model
        Model to test.
    test_dataloader : DataLoader
        The dataloader of testing data.

    Returns
    -------
    float
        Accuracy of the model in percentage.

    """
    with torch.no_grad():
        model.eval()

        for (xs, xs2, target) in test_dataloader:
            accuracy = Accuracy(task='multiclass', num_classes=7)
            
            output = model(xs, xs2)           
            acc = accuracy(output, target)
            
        return acc * 100
    
def plot_train_accuracy(train_logs):
    """Plot the training accuracy in a graph.

    Parameters
    ----------
    train_logs
        Training statistics of the models.

    """
    plt.figure()
    
    for log in train_logs:
        plt.plot(log.index, log.train_accuracy)

    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(['Linear', 'MLP', 'CNN', 'DCNN','CM1','CM2','CM3','DenseNet']);  
    
def plot_test_accuracy(models, test_dataloader):
    """Plot the testing accuracy in a graph.

    Parameters
    ----------
    models
        Models to test and plot the accuracy of.
    test_dataloader : DataLoader
        Testing Dataloader.

    """
    accuracy = []
    for model in models: 
        accuracy.append(test_model(model, test_dataloader))
        
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    plt.title('Testing Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    models = ['Linear', 'MLP', 'CNN', 'DCNN','CM1','CM2','CM3','DenseNet']
    
    max_y_lim = max(accuracy) + 0.5
    min_y_lim = min(accuracy) - 0.5
    plt.ylim(min_y_lim, max_y_lim)
    ax.bar(models, accuracy)
    plt.show()


def train(data: str, imgs: str):
    """Train models on given dataset.

    Parameters
    ----------
    data : str
        Data file path.
    imgs : str
        Image folder path.

    """
    
    # GPU Acceleration #
    device = torch.device('cuda:0' \
                          if torch.cuda.is_available() \
                          else 'cpu')
    print(device)
    
    print('Processing data...')
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(data, imgs)
    
    
    # Models #
    linear_model = models.LinearModel()
    mlp_model = models.MLPModel()
    cnn_model = models.CNNModel()
    dcnn_model = models.DCNNModel()
    cm1_model = models.CustomModel1()
    cm2_model = models.CustomModel2()
    cm3_model = models.CustomModel3()
    dense_net_model = models.DenseNet()
    
    # Train Models #
    print('Training models...')
    linear_model_log = train_model(linear_model, 30, 100, train_dataloader, val_dataloader)
    mlp_model_log = train_model(mlp_model, 30, 100, train_dataloader, val_dataloader)
    cnn_model_log = train_model(cnn_model, 30, 100, train_dataloader, val_dataloader)
    dcnn_model_log = train_model(dcnn_model, 30, 100, train_dataloader, val_dataloader)
    cm1_model_log = train_model(cm1_model, 30, 100, train_dataloader, val_dataloader)
    cm2_model_log = train_model(cm2_model, 45, 130, train_dataloader, val_dataloader)
    cm3_model_log = train_model(cm3_model, 45, 150, train_dataloader, val_dataloader)
    dense_net_model_log = train_model(dense_net_model, 45, 150, train_dataloader, val_dataloader)
    
    all_models = [linear_model, mlp_model, cnn_model, dcnn_model, cm1_model, 
                  cm2_model, cm3_model, dense_net_model]
    
    train_logs = [linear_model_log, mlp_model_log, cnn_model_log, dcnn_model_log,
            cm1_model_log, cm2_model_log, cm3_model_log, dense_net_model_log]
    
    # Plots #
    print('Plotting accuracy...')
    plot_train_accuracy(train_logs)
    plot_test_accuracy(all_models, test_dataloader)
    
    # Save Models #
    print('Saving models...')
    torch.save(linear_model, os.path.join(DIR, 'Prediction_Models/linear_model.pt'))
    torch.save(mlp_model, os.path.join(DIR, 'Prediction_Models/mlp_model.pt'))
    torch.save(cnn_model, os.path.join(DIR, 'Prediction_Models/cnn_model.pt'))
    torch.save(dcnn_model, os.path.join(DIR, 'Prediction_Models/dcnn_model.pt'))
    torch.save(cm1_model, os.path.join(DIR, 'Prediction_Models/cm1_model.pt'))
    torch.save(cm2_model, os.path.join(DIR, 'Prediction_Models/cm2_model.pt'))
    torch.save(cm3_model, os.path.join(DIR, 'Prediction_Models/cm3_model.pt'))
    torch.save(dense_net_model, os.path.join(DIR, 'Prediction_Models/dense_net_model.pt'))
    print('Done')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    