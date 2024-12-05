from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

def denormalize(tensor):
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    mean = torch.tensor(mean).view(1, 3, 1, 1) 
    std = torch.tensor(std).view(1, 3, 1, 1)    
    return tensor * std + mean

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),               
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    return transform


def get_transform_cifar():
    transform = transforms.Compose([
        transforms.ToTensor(),               
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])
    return transform

def load_MNIST_dataset(batch_size: int):
    transform = get_transform()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return train_loader


def load_CIFAR10_dataset(batch_size: int):
    transform = get_transform_cifar()
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return train_loader


