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

class Invoker:
    def __init__(self, device, sampler):
        self.receiver = Receiver(device, sampler)
        self.sender = Sender(self.receiver)
    
    async def generate(self):
        await self.sender.send()
    
    async def execute(self):
        await self.generate()



class Sender:
<<<<<<< Updated upstream
    def __init__(self, receiver, total_steps=1000, num_images=5):
=======
    def __init__(self, receiver, total_steps=1000, num_images=9):
>>>>>>> Stashed changes
        self.total_steps = total_steps
        self.receiver = receiver
        self.image_manager = ImageManager(num_images=num_images)

    async def send(self):
        images = [initialize_image(size=(28, 28)) for _ in range(len(self.image_manager.images))]
        for t in range(self.total_steps, 0, -1):
            print(f"Processing timestep {t}", end='\r')
            for idx in range(len(self.image_manager.images)):
                x_t = images[idx]
                x_t = await self.receiver.receive(x_t, t)
                images[idx] = x_t
                self.image_manager.update_image(idx, x_t, t)

            self.display_grid_cv2(images)

        print("\nGeneration completed. Showing final grid.")
        self.display_grid_cv2(images, final=True)

<<<<<<< Updated upstream
    def display_grid_cv2(self, images, final=False):
        """
        Wrapper for the OpenCV visualization function.
        """
        metadata = self.image_manager.get_metadata()
        show_images_cv2(images, metadata, scale_factor=9, final=final)
=======


    def save_progress_plot(self, images, timestep, mu, sigma, is_final=False):
        num_images = len(images)

        # Determine the grid size for subplots
        cols = int(num_images**0.5)
        rows = (num_images + cols - 1) // cols  # Ensure all images fit in the grid

        fig, axs = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=300)
        axs = axs.flatten() if num_images > 1 else [axs]  # Flatten in case of a single image

        for idx, img in enumerate(images):
            if isinstance(img, torch.Tensor):
                # Convert to numpy and move channels to the last dimension
                img = img.detach().cpu().numpy()
                img = img.transpose(1, 2, 0)

                # Normalize to [0, 1] range
                print(f"beofre normalization")
                print(f"img max: {img.max()}")
                print(f"img min: {img.min()}")
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                img = img.astype(np.float32)
                print("After normalization")
                print(f"img max: {img.max()}")
                print(f"img min: {img.min()}")


            # Display the image
            axs[idx].imshow(img, extent=(0, 32, 0, 32))  # Explicitly set dimensions
            axs[idx].axis('off')
            axs[idx].set_title(f'Image {idx + 1}')

        # Hide unused subplots if there are any
        for idx in range(num_images, len(axs)):
            axs[idx].axis('off')

        # Set the title and save the figure
        title = 'Final Generated Images' if is_final else f'Generation Progress - Timestep {timestep}'
        plt.suptitle(title)
        filename = 'final_result.png' if is_final else f'timestep_{timestep:04d}.png'
        plt.savefig(os.path.join('generation_progress', filename), format='png', dpi=300, bbox_inches='tight')
        plt.close()
>>>>>>> Stashed changes


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
<<<<<<< Updated upstream
        t_tensor = torch.tensor([t]).unsqueeze(0)
        eps_theta = self.model(x_t.view(1, -1), t_tensor)
        alpha_t = self.sampler.get_alpha(t_tensor)
        alpha_bar_t = self.sampler.get_alpha_bar_t(t_tensor)
        beta_t = self.sampler.linear_beta_schedueler(t_tensor)
        z = torch.randn_like(x_t) if t > 1 else 0
=======
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
>>>>>>> Stashed changes

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


def load_model(device, model_path="model_serialzed_350_epochs"):
    model = ScoreNetwork0().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


async def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with context_manager(
        batch_size=1000,
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
