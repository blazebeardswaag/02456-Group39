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
        t = self.sampler.sample_time_step()
        t = t.to(self.config.device)
        
        alpha_bar = self.sampler.get_alpha_bar_t(t).to(self.config.device) 
        eps = torch.randn_like(image, device=self.config.device)  
        img_noise, gen_noise = self.image_generator.sample_img_at_t(t, image, alpha_bar, eps)

        #img_noise  = img_noise.view(img_noise.shape[0], -1)
        pred_noise = self.unet(img_noise, t)
        #gen_noise = gen_noise.view(gen_noise.shape[0], -1)
        loss = self.compute_loss(pred_noise, gen_noise)

        self.optimizer.zero_grad()
        loss.backward()

       # print(f"Batch {batch_idx}, Gradients:")
       # for name, param in self.unet.named_parameters():
        #    if param.grad is not None:
         #       print(f"{name}: mean={param.grad.abs().mean().item():.6f}, "
          #              f"max={param.grad.abs().max().item():.6f}, "
           #             f"min={param.grad.abs().min().item():.6f}")
            #else:
             #   print(f"{name}: grad=None")

        # Gradient clipping
  
  
  
        torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_norm=self.clip_value)
        self.optimizer.step()
        return loss.item()


    def train(self, data_loader, num_epochs):

        # Reconfigure the dataloader for optimal GPU transfer
        data_loader = DataLoader(
            data_loader.dataset,
            batch_size=data_loader.batch_size,
            shuffle=True,
            num_workers=24,  # Use multiple CPU cores for data loading
            pin_memory=True,  # This is crucial for faster CPU->GPU transfer
            prefetch_factor=2,
            drop_last=True
        )
        self.config.num_epochs = num_epochs
        self.config.batch_size = data_loader.batch_size
        best_loss = float('inf')
        epochs_without_improvement = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                images, _ = batch
                loss = self.train_step(images, batch_idx)
                wandb.log({"batch loss": loss})
                epoch_loss += loss


            avg_loss = epoch_loss / len(data_loader)
            # log avg loss to wandb
            wandb.log({"loss": avg_loss, "epoch": epoch})

            #early stopping logic
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)
                wandb.save(self.config.MODEL_OUTPUT_PATH)
                wandb.log({"best_loss": best_loss, "epoch": epoch})
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    print(f"Stopping early due to no improvement in {self.patience} epochs.\n Best loss: {best_loss:.4f}")
                    break

            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")




        torch.save(self.unet.state_dict(), self.config.MODEL_OUTPUT_PATH)
        wandb.save(self.config.MODEL_OUTPUT_PATH)
        wandb.finish()
