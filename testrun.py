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
from data.preprocessor.data_handler import load_MNIST_dataset
import argparse


with context_manager(
    batch_size=96,
    LR=1e-2,
    experiment_name="mnist_training",
    scheduler_type="linear",
    device= torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
) as config: 

    train_loader = load_MNIST_dataset(config.batch_size)
    sampler = Sampler(config, config.batch_size)
    unet_model = UNet().to(config.device)
    image_generator = ImageGenerator(sampler, config.device)
    trainer = Trainer(unet=unet_model, config=config, sampler=sampler, image_generator=image_generator)
    trainer.train(train_loader, num_epochs=75) 



    # Inputs 
    # 1) data name
    # batch size 
    # LR
    # schedueler
    # steps 
    