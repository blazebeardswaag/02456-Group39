import torch.nn as nn 
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from models.unet import ScoreNetwork0 as UNet
from sampler.sampler import Sampler
from sampler.image_generator import ImageGenerator
from utils.image_saver import ImageSaver
import os

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Run `pip install wandb` to enable wandb logging, it's not not enabled!")

class Trainer(nn.Module):
    def __init__(self, unet, config, sampler, image_generator, lr=1e-3):
        super().__init__()
        self.sampler = sampler
        self.unet = unet
        self.optimizer = optim.AdamW(self.unet.parameters(), lr=config.LR if config.LR else lr, weight_decay=0.01)    
        self.loss_fn = nn.MSELoss()
        self.image_generator = image_generator
        self.config = config
        self.image_saver = ImageSaver()
        self.save_frequency = 100

        self.use_wandb = getattr(config, 'use_wandb', False)
        print(self.use_wandb)
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=getattr(config, 'wandb_project', 'default_project'),
                config=config.__dict__,
                resume="allow",
                mode='online' if getattr(config, 'wandb_mode', 'online') == 'online' else 'offline'
            )
            #sample_input = torch.randn(1, *self.unet.input_ shape).to(next(self.unet.parameters()))
            wandb.watch(self.unet, log="all", log_freq=10)


    def compute_loss(self, gen_noise, predicted_noise):
        return self.loss_fn(gen_noise, predicted_noise)

    def train_step(self, image, batch_idx):

        image = image.to(self.config.device)
        t = self.sampler.sample_time_step()
        t = t.to(self.config.device)
        
        alpha_bar = self.sampler.get_alpha_bar_t(t).to(self.config.device) 
        eps = torch.randn_like(image, device=self.config.device)  
        img_noise, gen_noise = self.image_generator.sample_img_at_t(t, image, alpha_bar, eps)

        # Flatten tensors and ensure they are on GPU
        flattened_x = img_noise.view(img_noise.shape[0], -1)
        gen_noise = gen_noise.view(gen_noise.size(0), -1)

        
        pred_noise = self.unet(flattened_x, t)

        # Compute loss
        loss = self.compute_loss(pred_noise, gen_noise)

        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return loss.item()

    def train(self, data_loader, num_epochs):
        self.config.num_epochs = num_epochs
        self.config.batch_size = data_loader.batch_size
        best_model_loss = 10
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                images, _ = batch
                loss = self.train_step(images, batch_idx)
                epoch_loss += loss

            avg_loss = epoch_loss / len(data_loader)
            if avg_loss< best_model_loss: 
                best_model_loss = avg_loss 
                torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)
                print(f"Best model loss: {best_model_loss}\nsaved to: {self.config.MODEL_OUTPUT_PATH}")

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

            # log avg loss to wandb
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.log({"epoch_loss": avg_loss, "epoch": epoch+1})


        torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.save(self.config.MODEL_OUTPUT_PATH)
            wandb.finish()
