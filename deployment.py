#!/usr/bin/env python
__author__ = 'Aaron Woodhouse'

"""
Deploy models on example deployment dataset.
"""

# Imports #
import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import (DataLoader, TensorDataset, Dataset)
from PIL import Image
import torchvision
from torchvision import transforms

DIR = os.path.dirname(os.path.realpath(__file__))
#PARENT = os.path.dirname(DIR)
sys.path.append(DIR)

from models import *

# Classes #    
class ImageDataset(Dataset):
    """Dataset of deployment images."""
    
    def __init__(self):
        """Initialize class.
        
        Attributes
        ----------
        _x_imgs : :float32:tensor
            Deployment image data.
        _x_data : :float32:tensor
            Deployment metadata.
        _y_labels : :long:tensor
            Deployment labels.

        """
        table_tensor, label_tensor, img_tensor = self.__preprocess_data()
        
        self._x_imgs = img_tensor.to(torch.float32)
        self._x_data = table_tensor.to(torch.float32)
        self._y_labels = label_tensor.to(torch.long)
    
    def __len__(self):
        """Number of samples in the dataset."""
        return len(self.y_labels)
    
    def __getitem__(self, idx):
        """Get deployment images, metadata, and labels at a given index."""
        return self.x_imgs[idx], self.x_data[idx], self.y_labels[idx]
    
    @property
    def x_imgs(self):
        """tensor: set deployment images."""
        return self._x_imgs
    
    @x_imgs.setter
    def x_imgs(self, new_val):
        self._x_imgs = new_val
    
    @x_imgs.deleter
    def x_imgs(self):
        del self._x_imgs
        
    @property
    def x_data(self):
        """tensor: set deployment data."""
        return self._x_data
    
    @x_data.setter
    def x_data(self, new_val):
        self._x_data = new_val
    
    @x_data.deleter
    def x_data(self):
        del self._x_data   
        
    @property
    def y_labels(self):
        """tensor: set deployment labels."""
        return self._y_labels
    
    @y_labels.setter
    def y_labels(self, new_val):
        self._y_labels = new_val
    
    @y_labels.deleter
    def y_train(self):
        del self._y_labels   
    
    
    def __preprocess_data(self):
        """Preprocess deployment data.
        
        Returns
        -------        
        table_tensor : tensor
            Deployment metadata.
        label_tensor : tensor
            Deployment labels.
        img_tensor : tensor
            Deployment image data.
        """
        
        # Table Data #
        table_data = pd.read_csv(os.path.join(DIR, 'Data/deployment_data.csv'))
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
        
        for x in table_data['image_id']:
            img = Image.open(os.path.join(DIR, 'Data/Deployment_Images/' + x + ".jpg"))
            
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
def test_model(model_file: str, display_imgs: bool, display_predictions: bool):
    """Test model on deployment data.

    Parameters
    ----------
    model_file : str
        Model file name.
    display_imgs : bool
        Display the images.
    display_predictions : bool
        Display the model predicitons.

    Returns
    -------
    acc : float
        Accuracy of the model in percentage.

    """
    # GPU Acceleration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    test_dataset = ImageDataset()
    model = load_model(model_file, device)
    dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    transform = transforms.ToPILImage()
    labels_map = ['bcc','bkl','df','nv','vasc','mel','akiec']
    
    stats = {}
    for x in labels_map:
        stats[x] = [0,0] # true positives / false positives
    
    acc = 0
    counter = 0
    with torch.no_grad():
        for xs, xs2, targets in dataloader:
            xs, xs2, targets = xs.to(device), xs2.to(device), targets.to(device)
            ys = model(xs, xs2)
            
            acc += (ys.argmax(axis=1) == targets).sum().item()
            
            if display_predictions:
                print("Img\tActual\tPredicted\tCorrect")
                
            for i in range(len(targets)):
                counter += 1
                x = labels_map[ys.argmax(axis=1)[i].item()]
                y = labels_map[targets[i].item()]
                c = (x == y)
                    
                if c: # true positives
                        stats[x][0] += 1
                else: # false positives
                        stats[x][1] += 1
                        
                if display_predictions:                        
                    print(f"{i}\t{y}\t{x}\t\t{str(c)[0]}")
                    
                if display_imgs:
                    img = transform(xs[i])
                    plt.figure()
                    plt.imshow(img)
                    plt.show()
            
            print("\nClass\tTP  FP")
            for (x,y) in stats.items():
                print(f"{x}\t{y}")
            
    acc = acc / counter * 100
    print("\nAccuracy = %.2f" % acc)
    
    return acc
    

def load_model(model_file: str, device):
    """Load the given model.

    Parameters
    ----------
    model_file : str
        Model file name.
    device
        Device to load model to.

    Returns
    -------
    model
        Loaded model instance.

    """
    print("Loading from " + model_file)
    model = torch.load(os.path.join(DIR, 'Prediction_Models/' + model_file)).to(device)
    return model    
    

def plot_test_accuracy(model_names):
    """Plot deployment test accuracy.

    Parameters
    ----------
    model_names : list
        Name of each model file.

    """
    accuracy = []
    for name in model_names: 
        accuracy.append(test_model(name, False, False))
        
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    
    plt.title('Testing Accuracy')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    model_names = ['Linear', 'MLP', 'CNN', 'DCNN','CM1','CM2','CM3','DenseNet']
    
    max_y_lim = max(accuracy) + 0.5
    min_y_lim = min(accuracy) - 0.5
    plt.ylim(min_y_lim, max_y_lim)
    ax.bar(model_names, accuracy)
    plt.show()    
    
def main():
    model_names = ['linear_model.pt', 'mlp_model.pt', 'cnn_model.pt',
                   'dcnn_model.pt', 'cm1_model.pt', 'cm2_model.pt', 'cm3_model.pt',
                   'dense_net_model.pt']
    
    print('Testing models...')
    plot_test_accuracy(model_names)
    print('Done')
    
if __name__ == '__main__':
    main()
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    