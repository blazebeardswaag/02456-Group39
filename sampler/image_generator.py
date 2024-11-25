import torch.nn as nn 
import torch 
from utils.helpers import get_alpha, linear_beta_schedueler, get_alpha_bar_t

class ImageGenerator:

    def __init__(self, sampler=None):
        self.sampler = sampler

    def reconstruct_image(self, x_t, predicted_noise, t,
                        alpha_t, alpha_bar_t, beta_t, z):
        alpha_t = get_alpha(t)
        std_t = beta_t 
        x_t_minus_one = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * z
        return x_t_minus_one



    def sample_single_image(self, t, eps_theta, sampler, xt):
            alpha_t = sampler.get_alpha(t)
            alpha_bar_t = sampler.get_alpha_bar_t(t)
            beta_t = sampler.linear_beta_schedueler(t)
            z = torch.randn_like(xt) if t > 1 else 0

            self.reconstruct_image(
                xt.view(28, 28),
                    eps_theta.view(28, 28), 
                    t, 
                    alpha_t, 
                    alpha_bar_t, 
                    beta_t, 
                    z)


    def sample_img_at_t(self, step, x_t, alpha_bar_t, eps):
        assert isinstance(x_t, torch.Tensor), f"x_t should be a tensor but got {type(x_t)} instead."
        assert torch.is_tensor(step) and step.dtype in [torch.int8, torch.int16, torch.int32, torch.int64], \
            f"step must be an integer tensor. step_dtype={step.dtype}"
        
        assert torch.all(step >= 1) and torch.all(step <= 1000), \
            "step must be integers between 1 and 999 (inclusive of 1, exclusive of 1000)."

        alpha_bar_t = alpha_bar_t.view(-1, 1, 1, 1)  # [batch_size, 1, 1, 1]
        
        x_t_plus_one = torch.sqrt(alpha_bar_t) * x_t + torch.sqrt(1 - alpha_bar_t) * eps
        return x_t_plus_one, eps


