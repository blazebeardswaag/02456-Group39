import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, sampler 
import torchvision.datasets as datasets
from torchvision import transforms 
from model_mnist import Network



#    weights = torch.load('MNIST_model.pth')
  #  model = Network()
  #    model.load_state_dict(weights)


def get_pred(model, images):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    proba = torch.exp(model(images)) 
    _, pred_label = torch.max(proba, dim=1)
    print(pred_label)
