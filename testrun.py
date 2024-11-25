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



with context_manager(
    batch_size=1000,
    LR=1e-4,
    experiment_name="mnist_training",
    scheduler_type="linear"
) as config: 

    train_loader = load_MNIST_dataset(config.batch_size)
    sampler = Sampler(config, config.batch_size)
    unet_model = UNet()
    image_generator = ImageGenerator(sampler)
    trainer = Trainer(unet=unet_model, config=config, sampler=sampler, image_generator=image_generator)
    trainer.train(train_loader, num_epochs=5) 



