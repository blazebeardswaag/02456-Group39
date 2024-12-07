import asyncio
import torch
from utils.helpers import (
    initialize_image,
    generate_one_step,
    show_images_cv2
)
from sampler.image_generator import ImageGenerator
from sampler.sampler import Sampler
from configs.config_manager import context_manager
from display.grid_display import ImageManager
from models.unet import ScoreNetwork0
import matplotlib.pyplot as plt 
import os
import cv2
import numpy as np

class Invoker:
    def __init__(self, device, sampler):
        self.receiver = Receiver(device, sampler)
        self.sender = Sender(self.receiver)
    
    async def generate(self):
        await self.sender.send()
    
    async def execute(self):
        await self.generate()



class Sender:
    def __init__(self, receiver, total_steps=1000, num_images=15, save_path="generated_images"):
        self.total_steps = total_steps
        self.receiver = receiver
        self.image_manager = ImageManager(num_images=num_images)
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)  # Ensure the directory exists

    async def send(self):
        images = [initialize_image(size=(28, 28)) for _ in range(len(self.image_manager.images))]
        for t in range(self.total_steps, 0, -1):
            print(f"Processing timestep {t}", end='\r')
            for idx in range(len(self.image_manager.images)):
                x_t = images[idx]
                x_t = await self.receiver.receive(x_t, t)
                images[idx] = x_t
                self.image_manager.update_image(idx, x_t, t)

            # Save images every 100 steps
            if t % 100 == 0 or t == 1:
                self.save_images(images, t)

            self.display_grid_cv2(images)

        print("\nGeneration completed. Showing final grid.")
        self.display_grid_cv2(images, final=True)

    def display_grid_cv2(self, images, final=False):
        """
        Wrapper for the OpenCV visualization function.
        """
        metadata = self.image_manager.get_metadata()
        show_images_cv2(images, metadata, scale_factor=9, final=final)

    def save_images(self, images, t):
        """
        Save the current grid of images to the specified folder.
        """
        # Convert each image tensor to a NumPy array
        images_np = [(img.detach().cpu().numpy() * 255).astype("uint8") for img in images]
        
        # Stack the images into a grid (adjust grid size as needed)
        grid_size = int(len(images)**0.5)  # Assuming a square grid
        rows = []
        for i in range(grid_size):
            row = np.hstack(images_np[i * grid_size: (i + 1) * grid_size])
            rows.append(row)
        grid = np.vstack(rows)
        
        # Save the grid as an image
        file_name = os.path.join(self.save_path, f"step_{t}_grid.png")
        cv2.imwrite(file_name, grid)
        print(f"Grid image saved for step {t} in {self.save_path}.")



class Receiver:
    def __init__(self, device, sampler):
        self.device = device
        self.model = load_model(device)
        self.sampler = sampler
        self.image_gen = ImageGenerator(self.sampler, device=self.device)

    async def receive(self, x_t, t):
        x_t = self.sample_one_step(x_t, t)
        return x_t

    def sample_one_step(self, x_t, t):
        t_tensor = torch.tensor([t], device=self.device).unsqueeze(0)
        eps_theta = self.model(x_t.view(1, -1), t_tensor)
        alpha_t = self.sampler.get_alpha(t_tensor)
        alpha_bar_t = self.sampler.get_alpha_bar_t(t_tensor)

        # Instead of calling cosine_beta_scheduler(t_tensor), just index into the precomputed schedule
        beta_t = self.sampler.beta_scheduler[t-1]

        z = torch.randn_like(x_t) if t > 1 else 0

        x_t = self.image_gen.reconstruct_image(
            x_t.view(28, 28),
            eps_theta.view(28, 28),
            t_tensor,
            alpha_t,
            alpha_bar_t,
            beta_t,
            z,
        )

        return x_t.view(28, 28)



def load_model(device, model_path="model_serialzed2"):
    model = ScoreNetwork0().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


async def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with context_manager(
        batch_size=1000,
        LR=2e-4,
        experiment_name="mnist_training",
        scheduler_type="linear",
        device=device
    ) as config:
        sampler = Sampler(config, config.batch_size, config.scheduler_type, 1000)
        invoker = Invoker(device, sampler)
        await invoker.execute()


if __name__ == "__main__":
    asyncio.run(main())
