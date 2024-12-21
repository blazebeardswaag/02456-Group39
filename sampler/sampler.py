import torch
import torch.nn as nn


class Sampler:
    def __init__(self, config, batch_size, rgb):
        self.is_rgb = rgb
        self.config = config
        self.dim = self.config.DIM
        self.batch_size = batch_size
        self.scheduler_type = "linear"

    def get_alpha_bar_t(self, t_tensor):
        t = int(t_tensor[0].item())

        alphas = torch.tensor([self.get_alpha(t_i) for t_i in range(1, t + 1)], dtype=torch.float32)
        alpha_bar_t = alphas.prod()

        alpha_bar_results = torch.full((t_tensor.size(0),), alpha_bar_t, dtype=torch.float32)

        return alpha_bar_results

    def get_alpha(self, t_tensor):
        beta_t = self.linear_beta_scheduler(t_tensor)
        alpha_t = 1 - beta_t
        return alpha_t

    def linear_beta_scheduler(self, step):
        d = torch.tensor((0.02 - 1e-4) / 1000, device=self.config.device)
        b_t = torch.tensor(1e-4, device=self.config.device) + step * d
        return b_t

    def sample_time_step(self):
        t = torch.randint(low=1, high=self.config.MAX_STEPS, size=(self.batch_size,1))
        return t