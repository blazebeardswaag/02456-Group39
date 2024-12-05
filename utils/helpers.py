import torch 
import numpy as np
from torchvision import datasets, transforms
from torch.utils import data
from PIL import Image
import cv2
from models.unet import ScoreNetwork0 
from sampler.image_generator import ImageGenerator
from sampler.sampler import Sampler 
from configs.config import Config 
from utils.image_saver import ImageSaver
from configs.config_manager import context_manager
import matplotlib.pyplot as plt


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


def load_model(device, model_path="model_serialzed"):

    model = ScoreNetwork0().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def initialize_image(size=(28, 28), channels=3):
    SIZE = (channels, *size)  
    eps = torch.normal(mean=0.0, std=1.0, size=SIZE)

    #x_t = torch.randn_like(SIZE) # [3, 32, 32] for RGB
    
    return eps

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
        print(f" min and max before scaling and normalizing min : {frame.min()}, max : {frame.max()}")
        print(f"frame shape: {frame.shape}")
        frame = ((frame + 1) / 2) * 255.0

        #frame = np.clip(frame, 0, 255).astype(np.uint8)
        print(f" min and max after scaling and normalizing min : {frame.min()}, max : {frame.max()}")

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
        color = (0, 0, 255) 
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

from torch.distributions.normal import Normal

def gaussian_discrete_decoder(x_continuous, mu, sigma, num_bins=256):
    C, H, W = x_continuous.shape
    x_centered = (x_continuous - mu) / (sigma + 1e-8)
    bin_width = 2.0 / num_bins
    bin_edges = torch.linspace(-1, 1 + bin_width, num_bins + 1).to(x_continuous.device)
    gaussian = Normal(x_centered.unsqueeze(-1), 1.0)
    cdf_values = gaussian.cdf(bin_edges)
    bin_probs = cdf_values[..., 1:] - cdf_values[..., :-1]
    x_discrete = torch.argmax(bin_probs, dim=-1)
    x_discrete = torch.clamp(x_discrete, 0, num_bins - 1)
    return x_discrete.to(torch.uint8)


def discrete_decoder(x_continuous, num_bins=256):
        
        x_scaled = (x_continuous + 1) * (num_bins - 1) / 2.0  # Scale to [0, 255]
        # Round to nearest discrete integer bin
        x_discrete = torch.round(x_scaled)
        # Clip values to ensure they're within [0, num_bins - 1]
        x_discrete = torch.clamp(x_discrete, min=0, max=num_bins - 1)

        return x_discrete.to(torch.uint8)  # Convert to uint8 for compatibility with image formats



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