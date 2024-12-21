import torch.nn as nn 
import torch 
from ..utils.diffusion_utils import sample_epsilon, get_alpha, linear_beta_schedueler, cosine_beta_scheduler, get_alpha_bar_t
from .sampler import Sampler 


class ImageGenerator:

    def __init__(self, sampler, device=None):
        self.sampler = sampler
        self.device = device if device is not None else sampler.config.device

    def reconstruct_image(self, x_t, predicted_noise, t, alpha_t, alpha_bar_t, beta_t, z):
        """
        Reconstruct image from noisy version.
        x_t: (B, C, H, W) where:
            - MNIST: C=1, H=W=28
            - CIFAR: C=3, H=W=32
        """
        # Move all tensors to the same device
        x_t = x_t.to(self.device)
        predicted_noise = predicted_noise.to(self.device)
        alpha_t = alpha_t.to(self.device)
        alpha_bar_t = alpha_bar_t.to(self.device)
        if torch.is_tensor(beta_t):
            beta_t = beta_t.to(self.device)
        if torch.is_tensor(z):
            z = z.to(self.device)

        # Ensure correct dimensions
        alpha_t = alpha_t.view(-1, 1, 1, 1)
        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)
        beta_t = beta_t.view(-1, 1, 1, 1) if torch.is_tensor(beta_t) else beta_t
        
        x_t_minus_one = (
            1 / torch.sqrt(alpha_t)
        ) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * z
        
        return x_t_minus_one

    def sample_img_at_t(self, step, x_t, alpha_bar_t, eps):
        """Sample image at specific timestep."""
        assert isinstance(x_t, torch.Tensor), f"x_t should be a tensor but got {type(x_t)} instead."
        assert torch.is_tensor(step) and step.dtype in [
            torch.int8, torch.int16, torch.int32, torch.int64
        ], f"step must be an integer tensor. step_dtype={step.dtype}"
        assert torch.all(step >= 1) and torch.all(step <= 1000), \
            "step must be integers between 1 and 999 (inclusive of 1, exclusive of 1000)."

        # Move tensors to correct device
        alpha_bar_t = alpha_bar_t.to(self.device).view(-1, 1, 1, 1)
        x_t = x_t.to(self.device)
        eps = eps.to(self.device)

        try:
            x_t_plus_one = torch.sqrt(alpha_bar_t) * x_t + torch.sqrt(1 - alpha_bar_t) * eps
        except Exception as e:
            print(f"Error: {e}")
            print(f"size of alpha_bar_t : {alpha_bar_t.shape}")
            print(f"size of x: {x_t.shape}")

        return x_t_plus_one, eps
