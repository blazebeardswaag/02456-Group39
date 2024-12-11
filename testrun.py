import torch.nn as nn 
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from models.unet_borrowed import Unet 
from sampler.sampler import Sampler
from trainer.trainer import Trainer
from configs.config import Config
from sampler.image_generator import ImageGenerator
from torchvision import datasets, transforms
from torch.utils.data import Subset
from configs.config_manager import context_manager
from data.preprocessor.data_handler import load_MNIST_dataset, load_CIFAR_dataset
import wandb
import argparse


with context_manager(
    batch_size=164,
    LR=2e-3,
    experiment_name="CIFAR_Training",
    scheduler_type="linear",
    use_wandb=True,
    MODEL_OUTPUT_PATH = "cry baby",
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
) as config:
    
    config_model =  {
        "im_channels": 3,
        "im_size": 32,
        "down_channels": [32, 64, 128, 256],
        "mid_channels": [256, 256, 128],
        "down_sample": [True, True, False],
        "time_emb_dim": 128,
        "num_down_layers": 2,
        "num_mid_layers": 2,
        "num_up_layers": 2,
        "num_heads": 4,
    }


    
    print("loading data")
    train_loader = load_CIFAR_dataset(config.batch_size, train=True)
    val_loader = load_CIFAR_dataset(config.batch_size, train=False)
    print("sampling data")
    sampler = Sampler(config, config.batch_size)
    unet_model = Unet(config_model).to(config.device)
    print(f"Model device: {next(unet_model.parameters()).device}")
    image_generator = ImageGenerator(sampler, config.device)
    trainer = Trainer(unet=unet_model, config=config, sampler=sampler, image_generator=image_generator)
    print("training")
    trainer.train(train_loader, val_loader, num_epochs=350)
    print("done training")