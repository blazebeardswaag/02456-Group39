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
import wandb
import argparse


with context_manager(
    batch_size=1024,
    LR=1e-3,
    experiment_name="mnist_training",
    scheduler_type="linear",
    use_wandb=True,

    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
) as config:
    print("loading data")

    train_loader = load_MNIST_dataset(config.batch_size)
    print("sampling data")
    sampler = Sampler(config, config.batch_size)
    unet_model = UNet().to(config.device)  # Model moved to GPU
    print(f"Model device: {next(unet_model.parameters()).device}")
    image_generator = ImageGenerator(sampler, config.device)
    trainer = Trainer(unet=unet_model, config=config, sampler=sampler, image_generator=image_generator)
    print("training")
    trainer.train(train_loader, num_epochs=75)
    print("done training")



    # Inputs 
    # 1) data name
    # batch size 
    # LR
    # schedueler
    # steps 
    