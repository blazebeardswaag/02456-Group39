import torch.nn as nn 
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from models.unet import ScoreNetwork0 as UNet
from sampler.sampler import Sampler
from sampler.image_generator import ImageGenerator
from utils.image_saver import ImageSaver
import os


class Trainer(nn.Module):
    def __init__(self, unet, config, sampler, image_generator, lr=1e-3):
        super().__init__()
        self.sampler = sampler
        self.unet = unet
        self.optimizer = optim.AdamW(self.unet.parameters(), lr)    
        self.loss_fn = nn.MSELoss()
        self.image_generator = image_generator
        self.config = config
        self.image_saver = ImageSaver()
        self.save_frequency = 100
        self.clip_value = 1.0

    def compute_loss(self, gen_noise, predicted_noise):
        return self.loss_fn(gen_noise, predicted_noise)

    def train_step(self, image, batch_idx):
        image = image.to(self.config.device)
        t = self.sampler.sample_time_step()
        print(f"shape t: {t.shape}")
        t = t.to(self.config.device)
        
        alpha_bar = self.sampler.get_alpha_bar_t(t).to(self.config.device) 
        eps = torch.randn_like(image, device=self.config.device)  
        img_noise, gen_noise = self.image_generator.sample_img_at_t(t, image, alpha_bar, eps)

        pred_noise = self.unet(img_noise, t)
        loss = self.compute_loss(pred_noise, gen_noise)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=self.clip_value)
        self.optimizer.step()
        return loss.item()

    def validation_step(self, image, batch_idx):
        self.unet.eval()
        with torch.no_grad():
            image = image.to(self.config.device)
            t = self.sampler.sample_time_step()
            t = t.to(self.config.device)
            
            alpha_bar = self.sampler.get_alpha_bar_t(t).to(self.config.device) 
            eps = torch.randn_like(image, device=self.config.device)  
            img_noise, gen_noise = self.image_generator.sample_img_at_t(t, image, alpha_bar, eps)

            pred_noise = self.unet(img_noise, t)
            loss = self.compute_loss(pred_noise, gen_noise)
        
        self.unet.train()
        return loss.item()

    def train(self, train_loader, val_loader, num_epochs):
        self.config.num_epochs = num_epochs
        self.config.batch_size = train_loader.batch_size
        best_model_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.unet.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                images, _ = batch
                loss = self.train_step(images, batch_idx)
                train_loss += loss

            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss = 0.0
            for batch_idx, batch in enumerate(val_loader):
                images, _ = batch
                loss = self.validation_step(images, batch_idx)
                val_loss += loss
                
            avg_val_loss = val_loss / len(val_loader)
            
            # Save best model based on validation loss
            if avg_val_loss < best_model_loss: 
                best_model_loss = avg_val_loss
                torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)
                print(f"Best model loss: {best_model_loss}\nsaved to: {self.config.MODEL_OUTPUT_PATH}")

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)