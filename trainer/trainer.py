import torch.nn as nn 
import torch
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset
from models.unet import ScoreNetwork0 as UNet
from sampler.sampler import Sampler
from sampler.image_generator import ImageGenerator
from utils.image_saver import ImageSaver
from torch.cuda.amp import GradScaler
from torchvision import transforms
import os

try:
    import wandb
except ImportError:
    print("Run `pip install wandb` to enable wandb logging, it's not enabled!")

class Trainer(nn.Module):
    def __init__(self, unet, config, sampler, image_generator, lr=1e-4):
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
        self.use_wandb = getattr(config, 'use_wandb', False)
        self.patience = 10
        print(self.use_wandb)

        wandb.init(
            project=getattr(config, 'wandb_project', 'cifar10'),
            config=config.__dict__,
            resume="allow",
            mode='online'
        )
        wandb.watch(self.unet, log="all", log_freq=10)

    def compute_loss(self, gen_noise, predicted_noise):
        return self.loss_fn(gen_noise, predicted_noise)

    def train_step(self, image, batch_idx):
        image = image.to(self.config.device)

        # Normalize with esoteric values
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.config.device)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=self.config.device)
        image = (image - mean[None, :, None, None]) / std[None, :, None, None]
        
        t = self.sampler.sample_time_step()
        t = t.to(self.config.device)
        
        alpha_bar = self.sampler.get_alpha_bar_t(t).to(self.config.device) 
        eps = torch.randn_like(image, device=self.config.device)  
        img_noise, gen_noise = self.image_generator.sample_img_at_t(t, image, alpha_bar, eps)

        pred_noise = self.unet(img_noise, t)
        loss = self.compute_loss(pred_noise, gen_noise)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=self.clip_value)
        self.optimizer.step()
        return loss.item()

    def validate(self, val_loader):
        self.unet.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(self.config.device)

                # Normalize the images
                mean = torch.tensor([0.4914, 0.4822, 0.4465], device=self.config.device)
                std = torch.tensor([0.2470, 0.2435, 0.2616], device=self.config.device)
                images = (images - mean[None, :, None, None]) / std[None, :, None, None]

                t = self.sampler.sample_time_step().to(self.config.device)
                alpha_bar = self.sampler.get_alpha_bar_t(t).to(self.config.device)
                eps = torch.randn_like(images, device=self.config.device)
                img_noise, gen_noise = self.image_generator.sample_img_at_t(t, images, alpha_bar, eps)

                pred_noise = self.unet(img_noise, t)
                loss = self.compute_loss(pred_noise, gen_noise)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        self.unet.train()  # Return the model to training mode
        return avg_val_loss

    def train(self, train_loader, val_loader, num_epochs):
        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=train_loader.batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            drop_last=True,
        )
        
        val_loader = DataLoader(
            val_loader.dataset,
            batch_size=val_loader.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=2,
            drop_last=True,
        )

        self.config.num_epochs = num_epochs
        self.config.batch_size = train_loader.batch_size
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                images, _ = batch
                loss = self.train_step(images, batch_idx)
                wandb.log({"batch loss": loss})
                epoch_loss += loss

            avg_train_loss = epoch_loss / len(train_loader)
            avg_val_loss = self.validate(val_loader)  # Evaluate on validation set

            #wandb.log({"train loss": avg_train_loss, "val loss": avg_val_loss, "epoch": epoch})

            # Early stopping logic
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)
                wandb.save(self.config.MODEL_OUTPUT_PATH)
                wandb.log({"train loss": avg_train_loss, "val loss": avg_val_loss, "epoch": epoch})
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Stopping early due to no improvement in {self.patience} epochs.\n Best val loss: {best_loss:.4f}")
                    break

            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)
        wandb.save(self.config.MODEL_OUTPUT_PATH)
        wandb.finish()