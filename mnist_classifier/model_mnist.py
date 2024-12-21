import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler 
import torchvision.datasets as datasets
from torchvision import transforms 

# Model definition
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # first hidden layer: 784 x 256
        self.hidden_0 = nn.Linear(784, 128)
        self.hidden_1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

        
    def forward(self, x):
        # flatten input to vector of 764
        x = x.view(x.shape[0], -1)
        # forward pass
        x = self.hidden_0(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.hidden_1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output(x)
        x = self.softmax(x)
        
        return x