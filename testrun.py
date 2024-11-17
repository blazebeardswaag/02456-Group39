import torch.nn as nn 
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from models.unet import ScoreNetwork0 as UNet
from sampler.sampler import Sampler
from trainer.trainer import Trainer
from configs.config import Config
from sampler.image_generator import ImageGenerator
from torchvision import datasets, transforms
from torch.utils.data import Subset
from configs.config_manager import context_manager


BATCH_SIZE = 20


# Normalize 
transform = transforms.Compose([
    transforms.ToTensor(),               
    transforms.Normalize((0.5,), (0.5,)) 
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
limited_indices = list(range(30000)) 
limited_dataset = Subset(train_dataset, limited_indices)
train_loader = DataLoader(limited_dataset, batch_size=BATCH_SIZE, shuffle=True)


with context_manager() as config: 
    sampler = Sampler(config, BATCH_SIZE)
    unet_model = UNet()
    image_generator = ImageGenerator(sampler)
    trainer = Trainer(unet=unet_model, config = config, sampler=sampler, lr=1e-3, image_generator=image_generator)

    trainer.train(train_loader, num_epochs=25)


