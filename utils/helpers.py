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


