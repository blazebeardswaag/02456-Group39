import torch.nn as nn 
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from models.unet import ScoreNetwork0 as UNet
from models.unet_borrowed import Unet
from sampler.sampler import Sampler
from trainer.trainer import Trainer
from configs.config import Config
from sampler.image_generator import ImageGenerator
from torchvision import datasets, transforms
from torch.utils.data import Subset
from configs.config_manager import context_manager
from data.preprocessor.data_handler import load_MNIST_dataset, load_CIFAR10_dataset
import argparse


with context_manager(
    batch_size=32,
    LR=1e-4,
    experiment_name="CIFAR_Training",
    scheduler_type="linear",
    use_wandb=True,
    device=torch.device("cuda")
) as config:
    
    config_model = {
        "im_channels": 3,
        "im_size": 32,
        "down_channels": [64, 128, 256],
        "mid_channels": [256, 512, 512],
        "down_sample": [True, True, False],
        "time_emb_dim": 128,
        "num_down_layers": 3,
        "num_mid_layers": 3,
        "num_up_layers": 3,
        "num_heads": 4,
    }

    train_loader = load_CIFAR10_dataset(config.batch_size)
    print("sampling data")
    sampler = Sampler(config, config.batch_size)
    model = Unet(config_model).to(config.device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
        model.to(config.device)
    print(f"Model device: {next(model.parameters()).device}")
    image_generator = ImageGenerator(sampler, config.device)
    trainer = Trainer(unet=model, config=config, sampler=sampler, image_generator=image_generator)
    print("training")
    trainer.train(train_loader, num_epochs=1500)
    print("done training")

    