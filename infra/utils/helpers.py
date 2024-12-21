import torch 
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch.utils import data
from PIL import Image
import cv2
import os 
from ..models import ScoreNetwork0 
from ..configs.config import Config 
from ..utils.image_saver import ImageSaver
from ..configs.config_manager import context_manager
from .diffusion_utils import sample_epsilon, get_alpha, linear_beta_schedueler
from ..sampler import Sampler

def generate_one_step(model, sampler, t, image_generator, x_t ):
            t_tensor = torch.tensor([t]).unsqueeze(0)
            eps_theta = model(x_t.view(1, -1), t_tensor)
            alpha_t = sampler.get_alpha(t_tensor)
            alpha_bar_t = sampler.get_alpha_bar_t(t_tensor)
            beta_t = sampler.linear_beta_schedueler(t_tensor)
            z = torch.randn_like(x_t) if t > 1 else 0

            x_image = image_generator.reconstruct_image(
                x_t.view(28, 28),
                eps_theta.view(28, 28),
                t_tensor,
                alpha_t,
                alpha_bar_t,
                beta_t,
                z,
            )
            x_image = x_image.view(28, 28)

            return x_image


def load_model(device, model_path):

    model = ScoreNetwork0().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def initialize_image(size=(28, 28)):

        SIZE = torch.zeros(size)
        x_t = torch.randn_like(SIZE)
        x_t = (x_t + 1) / 2  # Normalize to [0, 1]

        return x_t


def generate_single_image(image_idx, model, sampler, x_t):
 
    image_generator = ImageGenerator()

    with torch.no_grad():
        for t in range(1000, 1, -1):
            print(f"Constructing image {image_idx + 1} at timestep: {t}", end="\r")
            t_tensor = torch.tensor([t]).unsqueeze(0)

            # Calculate required values
            eps_theta = model(x_t.view(1, -1), t_tensor)
            alpha_t = sampler.get_alpha(t_tensor)
            alpha_bar_t = sampler.get_alpha_bar_t(t_tensor)
            beta_t = sampler.linear_beta_schedueler(t_tensor)
            z = torch.randn_like(x_t) if t > 1 else 0

            x_t = image_generator.reconstruct_image(
                x_t.view(28, 28),
                eps_theta.view(28, 28),
                t_tensor,
                alpha_t,
                alpha_bar_t,
                beta_t,
                z,
            )
            x_t = x_t.view(28, 28)

    return x_t


def show_images_cv2(frames, metadata, scale_factor=9, final=False):
  
    processed_frames = []
    for idx, frame in enumerate(frames):
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().numpy()

        # Normalize to [0, 1] and then scale to [0, 255]
        frame = ((frame + 1) / 2) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        # Ensure grayscale image
        if frame.ndim == 2:  # Single-channel grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to BGR for OpenCV overlay

        height, width = frame.shape[:2]
        frame = cv2.resize(
            frame,
            (width * scale_factor, height * scale_factor),
            interpolation=cv2.INTER_NEAREST,
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (255, 255, 255) 
        thickness = 1
        position = (10, 20)

        cv2.putText(
            frame,
            f'Pos: {metadata[idx]["position"]}, t: {metadata[idx]["t"]}',
            position,
            font,
            font_scale,
            color,
            thickness,
        )

        processed_frames.append(frame)

    # Create the image grid
    grid = create_image_grid(processed_frames, cols=5)

    # Display the grid using OpenCV
    cv2.imshow("Denoising Process Grid", grid)

    # Manage OpenCV window state
    if final:
        while True:
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
    else:
        if cv2.waitKey(10) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            raise KeyboardInterrupt("Visualization interrupted by user.")





def create_image_grid(images, cols):
    rows = (len(images) + cols - 1) // cols
    height, width, channels = images[0].shape
    grid = np.zeros((rows * height, cols * width, channels), dtype=np.uint8)

    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        grid[row * height:(row + 1) * height, col * width:(col + 1) * width, :] = img

    return grid


def show_image(frame, image_number, scale_factor=5):
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    frame = (frame * 255).astype(np.uint8) if frame.dtype != np.uint8 else frame

    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (width * scale_factor, height * scale_factor), interpolation=cv2.INTER_NEAREST)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (255, 255, 255)
    thickness = 2
    position = (10, 30)
    cv2.putText(frame, f'Image #: {image_number}', position, font, font_scale, color, thickness)

    cv2.imshow('Denoising Process', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        raise KeyboardInterrupt("Visualization interrupted by user.")



def save_images(task: str, images: list, timestep: int, rgb: bool = False) -> None:
   
    main_dir = f"sampled_images_{task}"
    os.makedirs(main_dir, exist_ok=True)

    # Save only if timestep <= 50 or if it's a multiple of 10
    if timestep > 50 and timestep % 10 != 0:
        return

    # Save each image in its own folder within the main directory
    for idx, img in enumerate(images):
        # Create directory for this specific image if it doesn't exist
        image_dir = os.path.join(main_dir, f"image_{idx}")
        os.makedirs(image_dir, exist_ok=True)
        
        # Convert tensor to PIL Image
        img = img.cpu().detach()
        img = (img * 255).to(torch.uint8)
        
        # Handle RGB vs grayscale
        if rgb:
            img = img.permute(1, 2, 0).numpy() if img.dim() == 3 else img.numpy()
            mode = 'RGB'
        else:
            img = img.numpy()
            mode = 'L'
            
        img = Image.fromarray(img, mode=mode)
        
        # Save image with timestep in filename
        img.save(os.path.join(image_dir, f"step_{timestep:04d}.png"))
    
    if timestep % 10 == 0 or timestep <= 50:
        print(f"Saved images at timestep {timestep}")




############################


def initialize_batch(num_images, image_size, rgb=False):
    channels = 3 if rgb else 1
    return torch.randn(num_images, channels, *image_size)

def process_timestep(x_t, t, model, sampler, image_gen, num_images, image_size, rgb):
    channels = 3 if rgb else 1
    device = x_t.device 
    if rgb:
        t_tensor = torch.tensor([t], dtype=torch.long).to(device)
    else:
        t_tensor = torch.full((x_t.size(0), 1), t, dtype=torch.float32, device=device)

    if rgb:
        model_input = x_t  
    else:
        model_input = x_t.view(x_t.size(0), -1)

    eps_theta = model(model_input, t_tensor)
    
    alpha_t = sampler.get_alpha(t_tensor).to(device)
    alpha_bar_t = sampler.get_alpha_bar_t(t_tensor).to(device)
    beta_t = sampler.linear_beta_scheduler(t).to(device)
    z = torch.randn_like(x_t).to(device) if t > 1 else 0
    
    if not rgb:
        eps_theta = eps_theta.view(num_images, channels, *image_size)
    
    x_t = image_gen.reconstruct_image(
        x_t,
        eps_theta,
        t_tensor,
        alpha_t,
        alpha_bar_t,
        beta_t,
        z
    )
    
    return x_t

def setup_plot(num_images: int) -> tuple:
    """Setup matplotlib plot for visualization."""
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))
    if num_images == 1:
        axes = [axes]
    return fig, axes

def update_plot(axes, x_t: torch.Tensor, t: int, rgb: bool = False) -> None:
    """Update plot with new images."""
    for idx in range(len(axes)):
        img = x_t[idx].cpu().detach()
        axes[idx].clear()
        
        if rgb:
            img = img.permute(1, 2, 0).numpy()
        else:
            img = img.numpy()
            
        axes[idx].imshow(img, cmap='gray' if not rgb else None)
        axes[idx].axis('off')
        axes[idx].set_title(f'Image {idx + 1}')
    
    plt.suptitle(f'Timestep {t}')
    plt.tight_layout()
    plt.show()


def show_diffusion_process(device, model, num_images: int, image_size: tuple, rgb: bool = False) -> None:
    import matplotlib.pyplot as plt
    from IPython.display import clear_output, display
    from ..sampler import ImageGenerator
    import numpy as np
    
        
    device = torch.device(device)
    model = model.to(device)
    n_cols = int(np.ceil(np.sqrt(num_images)))
    n_rows = int(np.ceil(num_images / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.15, n_rows * 1.25))  # Adjusted size
    axes = axes.flatten() if num_images > 1 else [axes]
    
    # Normalization parameters for CIFAR-like datasets
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device) if rgb else torch.tensor([0.5], device=device)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device) if rgb else torch.tensor([0.5], device=device)
    
    with context_manager(experiment_name="mnist_generation", device=device) as config:
        # Setup
        sampler = Sampler(config, batch_size=num_images, rgb=rgb)        
        x_t = initialize_batch(num_images, image_size, rgb).to(device)
        image_gen = ImageGenerator(sampler, device)
        
        # Show 8 key timesteps
        key_timesteps = [1000, 875, 750, 625, 500, 375, 250, 125, 1]
        
        with torch.no_grad():
            for t in range(1000, 0, -1):
                x_t = process_timestep(x_t, t, model, sampler, image_gen, num_images, image_size, rgb)
                
                if t in key_timesteps:
                    # Normalize and convert to numpy
                    imgs = x_t * std[:, None, None] + mean[:, None, None]
                    imgs = imgs.cpu().numpy()
                    
                    # Update plots
                    for idx, ax in enumerate(axes):
                        ax.clear()
                        if idx < num_images:
                            img = imgs[idx]
                            if rgb:
                                img = img.transpose(1, 2, 0)  # CHW -> HWC
                            else:
                                img = img.squeeze(0)
                            
                            img = np.clip(img, 0, 1)  # Clip to [0, 1] range
                            ax.imshow(img, cmap='gray' if not rgb else None)
                        ax.axis('off')
                    
                    plt.suptitle(f'Timestep {t}')
                    plt.tight_layout()
                    
                    display(plt.gcf())
                    if t > 1:
                        clear_output(wait=True)
    
    plt.close()

def sample_images(device, model, num_images: int, image_size: tuple, rgb: bool = False, output: str = None) -> None:
    """
    Sample images using the same logic as show_diffusion_process.
    Saves images individually in task-specific folders.
    """
    from ..sampler import ImageGenerator  
    
    if not torch.cuda.is_available() and device == "cuda":
        print("CUDA not available, using CPU instead")
        device = "cpu"
    
    device = torch.device(device)
    model = model.to(device)
    
    # Create output directory
    task = "cifar" if rgb else "mnist"
    output_dir = f"generated_images/{output if output else task}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Standard normalization parameters
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=device) if rgb else torch.tensor([0.5], device=device)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=device) if rgb else torch.tensor([0.5], device=device)
    
    with context_manager(experiment_name="mnist_generation", device=device) as config:
        # Setup
        sampler = Sampler(config, batch_size=num_images, rgb=rgb)        
        x_t = initialize_batch(num_images, image_size, rgb).to(device)
        image_gen = ImageGenerator(sampler, device)
        
        # Sample using the same process as show_diffusion
        with torch.no_grad():
            for t in range(1000, 0, -1):
                x_t = process_timestep(x_t, t, model, sampler, image_gen, num_images, image_size, rgb)
        
        # Save individual images
        for i in range(num_images):
            img = x_t[i]
            if rgb:
                img = img.permute(1, 2, 0)
            else:
                img = img.squeeze(0)
            
            # Convert to numpy and ensure proper range [0, 255]
            img = img.cpu().numpy()
            img = (img * 255).clip(0, 255).astype(np.uint8)
            
            # Save image
            img_path = os.path.join(output_dir, f'sample_{i+1}.png')
            if rgb:
                Image.fromarray(img).save(img_path)
            else:
                Image.fromarray(img, mode='L').save(img_path)
