from infra import Trainer 
from typing import Optional # Lets figure out first what types we use 
import torch.nn as nn 
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from infra import ScoreNetwork0, Unet
from infra.sampler import Sampler
from infra.sampler import ImageGenerator
from infra.utils.image_saver import ImageSaver
import os
from infra.utils.helpers import sample_images, save_images, show_diffusion_process

# unet, config, sweep_config, sampler, image_generator


class DenoisingDiffusion:
    def __init__(self, task:str, device):
        self.device = device
        assert task.lower() in ["mnist","cifar"], "Please choose between mnist or cifar"
        self.task = task.lower()
        self.image_size = (32, 32) if self.task == "cifar" else (28, 28)
        if task.lower() == "mnist":
            self.model = self.load_model_weights(ScoreNetwork0(), self.device, self.task)
        else:
            self.model = self.load_CIFAR_unet()
        self.model = self.model.to(self.device)

    def load_model_weights(self, model, device, model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        return model


    def load_CIFAR_unet(self): 
        config_model =  {
                    "im_channels": 3,
                    "im_size": 32,
                    "down_channels": [32, 64, 128, 256],
                    "mid_channels": [256, 256, 128],
                    "down_sample": [True, True, False],
                    "time_emb_dim": 128,
                    "num_down_layers": 4,
                    "num_mid_layers": 2,
                    "num_up_layers": 4,
                    "num_heads": 4,
                }      
        
        model = Unet(config_model)
        return self.load_model_weights(model, self.device, "cifar")


    def sample_images(self, num_images, output_dir) -> None:
        rgb = self.task == "cifar"
        sample_images(self.device, self.model, num_images, self.image_size, rgb=rgb, output=output_dir)
    

    def show_diffusion(self, num_images=5):
        rgb = self.task == "cifar"
        print(f"rgb is: {rgb}")
        show_diffusion_process(self.device, self.model, num_images, self.image_size, rgb=rgb)


    def train(self, channels: int): 
        assert channels in [2,3], "Please choose 2 for greyscale or 3 channels for rgb."
        return 
    