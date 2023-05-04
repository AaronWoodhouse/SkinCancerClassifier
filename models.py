__author__ = 'Aaron Woodhouse'

"""
Prediciton Models.
"""

# Imports #
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision

class LinearModel(torch.nn.Module):    
    def __init__(self):
        super().__init__()

        self.flat = nn.Flatten()
        self.output = nn.LazyLinear(7)

    def forward(self, img, data):
        x1 = img
        x2 = data
        
        x1 = self.flat(x1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.output(x)
        return x
    
class MLPModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.flat = nn.Flatten()
        self.ll = nn.LazyLinear(200)
        self.relu = nn.ReLU()
        self.output = nn.LazyLinear(7)

    def forward(self, img, data):
        x1 = img
        x2 = data
        
        x1 = self.flat(x1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.ll(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x
    
class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 5, 3, padding='same'),
            nn.MaxPool2d(2),
        )
        
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.ll = nn.LazyLinear(200)
        self.output = nn.LazyLinear(7)

    def forward(self, img, data):
        x1 = img
        x2 = data
        
        x1 = self.conv(x1)
        x1 = self.relu(x1)
        
        x1 = self.flat(x1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.ll(x)
        x = self.relu(x)
        
        x = self.output(x) 
        return x        
    
class DCNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 10, 5, padding='same'),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 3, padding='same'),
            nn.MaxPool2d(2),
        )
        
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.ll = nn.LazyLinear(200)
        self.output = nn.LazyLinear(7)

    def forward(self, img, data):
        x1 = img
        x2 = data
        
        x1 = self.conv(x1)
        x1 = self.relu(x1)
        
        x1 = self.flat(x1)    
        x = torch.cat((x1, x2), dim=1)
        
        x = self.ll(x)
        x = self.relu(x)
        
        x = self.output(x)
        return x
    
class CustomModel1(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 10, 5, padding='same'),
            nn.MaxPool2d(2),
            nn.Conv2d(10, 20, 5, padding='same'),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 5, padding='same'),
            nn.MaxPool2d(3),            
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
        )
        
        self.relu = nn.ReLU()
        self.flat = nn.Flatten()
        self.ll = nn.LazyLinear(200)
        self.output = nn.LazyLinear(7)

    def forward(self, img, data):        
        x1 = img
        x2 = data
        
        x1 = self.conv(x1)
        x1 = self.relu(x1)
        
        x1 = self.flat(x1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.ll(x)
        
        res = x
        x = self.mlp(x) + res
        x = self.mlp(x) + res
        x = self.relu(x)
        
        x = self.output(x)
        return x
    
class CustomModel2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 25, 5, padding='same'),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(25, 50, 3, padding='same'),
            nn.MaxPool2d(3),
            nn.ReLU(),
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
        )
        
        self.flat = nn.Flatten()
        self.ll = nn.LazyLinear(200)
        self.output = nn.LazyLinear(7)

    def forward(self, img, data):        
        x1 = img
        x2 = data
        
        x1 = self.conv(x1)
        
        x1 = self.flat(x1)
        x = torch.cat((x1, x2), dim=1)
        
        x = self.ll(x)
        
        res = x
        x = self.mlp(x) + res
        x = self.mlp(x) + res
        
        x = self.output(x)
        return x
    
class CustomModel3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 25, 5, padding='same'),
            nn.MaxPool2d(3),
            nn.ReLU(),
            nn.Conv2d(25, 50, 3, padding='same'),
            nn.MaxPool2d(3),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(800, 200, batch_first=True, num_layers=3) # properties to remember for each, # 50 params for each input vector
        
        self.mlp = nn.Sequential(
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
        )
        
        self.flat = nn.Flatten()
        self.ll = nn.LazyLinear(200)
        self.output = nn.LazyLinear(7)

    def forward(self, img, data):        
        x1 = img
        x2 = data
        
        x1 = self.conv(x1)
        x1 = self.flat(x1)
        
        y, (h, c) = self.lstm(x1)
        x = torch.cat((y, x2), dim=1)
        
        x = self.ll(x)
        
        res = x
        x = self.mlp(x) + res
        x = self.mlp(x) + res
        
        x = self.output(x)
        return x

class DenseNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        preloaded = torchvision.models.densenet121(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(3, 64, 7, 2, 3)
        self.classifier = nn.Linear(1024+3, 7)
        
    def forward(self, img, data):
        x1 = img
        x2 = data
        features = self.features(x1)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = torch.cat((out, x2), dim=1)
        out = self.classifier(out)
        return out