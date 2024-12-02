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
import pprint


def train(config=None):
    with wandb.init(
        project='default_project',
        config = config,
        resume="allow"
    ):
        config = wandb.config
        print(f"config: {config}")
        

        train_loader = load_MNIST_dataset(config.batch_size)
        sampler = Sampler(old_config, config.batch_size, config.scheduler_type, config.MAX_STEPS)
        unet_model = UNet().to(old_config.device)  # Model moved to GPU
        image_generator = ImageGenerator(sampler, old_config.device)
        trainer = Trainer(unet=unet_model, config=old_config, sweep_config=config, sampler=sampler, image_generator=image_generator)
        trainer.train(train_loader)


with context_manager(
    experiment_name="mnist_training",
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
) as old_config:

    #print(f"config.sweep_config: {config.sweep_config}")
    #pprint.pprint(config.sweep_config)
    #print(config['parameters'])
    sweep_id = wandb.sweep(old_config.sweep_config, project="default_project")

    wandb.agent(sweep_id, train)