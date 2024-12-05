import asyncio
import torch
from utils.helpers import (
    initialize_image,
    generate_one_step,
    show_images_cv2,
    gaussian_discrete_decoder,
    discrete_decoder
)
from torch.amp import autocast
from data.preprocessor.data_handler import denormalize, get_transform_cifar
from sampler.image_generator import ImageGenerator
from sampler.sampler import Sampler
from configs.config_manager import context_manager
from display.grid_display import ImageManager
from models.unet import ScoreNetwork0
from models.unet_borrowed import Unet 
import matplotlib.pyplot as plt 
import os 
import sys
import numpy as np
import cv2

class Invoker:
    def __init__(self, device, sampler):
        self.receiver = Receiver(device, sampler)
        self.sender = Sender(self.receiver)
    
    async def generate(self):
        await self.sender.send()
    
    async def execute(self):
        await self.generate()



class Sender:
    def __init__(self, receiver, total_steps=1000, num_images=1):
        self.total_steps = total_steps
        self.receiver = receiver
        self.image_manager = ImageManager(num_images=num_images)
        os.makedirs('generation_progress', exist_ok=True)

    async def send(self):
        images = [initialize_image(size=(32, 32)) for _ in range(len(self.image_manager.images))]
        mu = torch.tensor(0, device='cpu')
        sigma = torch.tensor(1, device='cpu')
        
        for t in range(self.total_steps, 0, -1):
            print(f"Processing timestep {t}", end='\r')
            for idx in range(len(self.image_manager.images)):
                x_t = images[idx]
                x_t = await self.receiver.receive(x_t, t)
                images[idx] = x_t
                self.image_manager.update_image(idx, x_t, t)

            if t % 100 == 0 or t == self.total_steps or t == 1:
                mu = torch.stack(images).mean()
                sigma = torch.stack(images).std()
                self.save_progress_plot(images, t, mu, sigma)

        print("\nGeneration completed.")
        self.save_progress_plot(images, 0, mu, sigma, is_final=True)

    def save_progress_plot(self, images, timestep, mu, sigma, is_final=False):
        plt.figure(figsize=(2, 2))  # Small figure size since images are tiny
        for idx, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                img = img.detach().numpy()
                img = img.transpose(1, 2, 0)  # CHW -> HWC
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                
                plt.subplot(1, len(images), idx + 1)
                plt.imshow(img, interpolation='nearest')  
                plt.axis('off')
                
        title = 'Final Generated Images' if is_final else f'Generation Progress - Timestep {timestep}'
        plt.suptitle(title, fontsize=8)

        filename = 'final_result.png' if is_final else f'timestep_{timestep:04d}.png'
        plt.savefig(os.path.join('generation_progress', filename), 
                    format='png', 
                    dpi=300,  # Standard high-quality DPI
                    bbox_inches='tight',
                    pad_inches=0.1)
        plt.close()
class Receiver:
    def __init__(self, device, sampler):
        self.device = device
        self.model = load_model(device)
        self.sampler = sampler
        self.image_gen = ImageGenerator(self.sampler)

    async def receive(self, x_t, t):
        x_t = self.sample_one_step(x_t, t)
        return x_t

    def sample_one_step(self, x_t, t):
        t_tensor = torch.tensor(t, device='cpu').reshape(1,)
        x_t = x_t.to('cpu')
        
        with torch.no_grad():
            eps_theta = self.model(x_t.unsqueeze(0), t_tensor)
            
            alpha_t = self.sampler.get_alpha(t_tensor)
            alpha_bar_t = self.sampler.get_alpha_bar_t(t_tensor)
            beta_t = self.sampler.linear_beta_schedueler(t_tensor)
            z = torch.randn_like(x_t) if t > 1 else 0
            
            x_t_next = self.image_gen.reconstruct_image(
                x_t.view(3, 32, 32),
                eps_theta.view(3, 32, 32),
                t_tensor,
                alpha_t,
                alpha_bar_t,
                beta_t,
                z
            )

        if x_t_next.isnan().any():
            print(f"\n[First NaN detected in reconstruction at timestep {t}]")
            print(f"Coefficients:")
            print(f"  alpha_t: {alpha_t.item():.6f}")
            print(f"  alpha_bar_t: {alpha_bar_t.item():.6f}")
            print(f"  beta_t: {beta_t.item():.6f}")
            print(f"Input x_t stats: min={x_t.min():.6f}, max={x_t.max():.6f}")
            print(f"eps_theta stats: min={eps_theta.min():.6f}, max={eps_theta.max():.6f}")
            print(f"Output x_t_next stats: min={x_t_next.min():.6f}, max={x_t_next.max():.6f}")
            return x_t

        return x_t_next.view(3, 32, 32)

def load_model(device, model_path="./CIFAR_Transform"):
    import os
    print("model called")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    try:
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
 
        model = Unet(config_model).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

    return model


async def main():
    device = torch.device("cpu")

    with context_manager(
        batch_size=1,
        LR=1e-4,
        experiment_name="mnist_training",
        scheduler_type="linear",
        device=device
    ) as config:
        sampler = Sampler(config, batch_size=1)
        invoker = Invoker(device, sampler)
        await invoker.execute()


if __name__ == "__main__":
    asyncio.run(main())
