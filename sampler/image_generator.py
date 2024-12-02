import torch.nn as nn 
import torch 
from utils.helpers_model import sample_epsilon, get_alpha, linear_beta_schedueler, cosine_beta_scheduler, get_alpha_bar_t

class ImageGenerator:

    def __init__(self, sampler=None, device=None):
        self.sampler = sampler
        self.device = device

    def reconstruct_image(self, x_t, predicted_noise, t, alpha_t, alpha_bar_t, beta_t, z):
        alpha_t = get_alpha(t)
        std_t = beta_t
        x_t_minus_one = (
            1 / torch.sqrt(alpha_t)
        ) * (
            x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        ) + torch.sqrt(beta_t) * z
        return x_t_minus_one

    def sample_img_at_t(self, step, x_t, alpha_bar_t, eps):
        assert isinstance(x_t, torch.Tensor), f"x_t should be a tensor but got {type(x_t)} instead."
        assert torch.is_tensor(step) and step.dtype in [
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ], f"step must be an integer tensor. step_dtype={step.dtype}"

        assert torch.all(step >= 1) and torch.all(
            step <= 1000
        ), "step must be integers between 1 and 999 (inclusive of 1, exclusive of 1000)."

        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1).to(self.device)  # [batch_size, 1, 1, 1]
        x_t = x_t.to(self.device)
        eps = eps.to(self.device)
        try:
            x_t_plus_one = torch.sqrt(alpha_bar_t) * x_t + torch.sqrt(1 - alpha_bar_t) * eps
        except Exception as e:
            print(f"Error: {e}")
            print(f"size of alpha_bar_t : {alpha_bar_t.shape}")
            print(f"size of x: {x_t.shape}")

        return x_t_plus_one, eps
