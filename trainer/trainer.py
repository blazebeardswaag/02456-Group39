import torch.nn as nn 
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from models.unet import ScoreNetwork0 as UNet
from sampler.sampler import Sampler
from sampler.image_generator import ImageGenerator
from utils.image_saver import ImageSaver
import os

### TODO: Use EarlyStopping and train until convergance

class Trainer(nn.Module):
    def __init__(self, unet, config, sampler, image_generator, lr=1e-3):
        super().__init__()
        self.sampler = sampler
        self.unet = unet
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.image_generator = image_generator
        self.config = config
        self.image_saver = ImageSaver()
        self.save_frequency = 100  

    def compute_loss(self, gen_noise, predicted_noise):
        return self.loss(gen_noise, predicted_noise)  
    
    def train_step(self, image, batch_idx):
        t = self.sampler.sample_time_step()
        
        alpha_bar = self.sampler.get_alpha_bar_t(t)  
        eps = torch.randn_like(image)  
        img_noise, gen_noise = self.image_generator.sample_img_at_t(t, image, alpha_bar, eps)

        if batch_idx % 10 == 0:
            """         
            for img_idx in range(min(4, image.shape[0])):
                            self.image_saver.save_image_pair(
                                original_img=image[img_idx:img_idx+1],
                                noised_img=img_noise[img_idx:img_idx+1],
                                timestep=t[img_idx].item(),
                                batch_idx=f"{batch_idx}_img{img_idx}"
                            )
            """
        flattened_x = img_noise.view(img_noise.shape[0], -1)
        gen_noise = gen_noise.view(gen_noise.size(0), -1)
        pred_noise = self.unet(flattened_x, t)
        loss = self.compute_loss(pred_noise, gen_noise)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, data_loader, num_epochs):
        for epoch in range(num_epochs):
            epoch_loss = 0.0 
            for batch_idx, batch in enumerate(data_loader):
                images, _ = batch
                loss = self.train_step(images, batch_idx)
                epoch_loss += loss

            avg_loss = epoch_loss / len(data_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)
