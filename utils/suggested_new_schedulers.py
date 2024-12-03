import torch
import math


class sampler:
    def __init__(self, config, batch_size, scheduler_type, MAX_STEPS):
        self.config = config
        self.dim = self.config.DIM
        self.batch_size = batch_size
        self.scheduler_type = scheduler_type
        self.MAX_STEPS = MAX_STEPS

        if self.scheduler_type == "linear":
            self.beta_scheduler = self.linear_beta_scheduler(self.MAX_STEPS)
        elif self.scheduler_type == "cosine":
            self.beta_scheduler = self.cosine_beta_scheduler(self.MAX_STEPS)
        
        self.alpha_scheduler = 1 - self.beta_scheduler
        self.alpha_bar_scheduler = torch.cumprod(self.alpha_scheduler, dim=0).to(self.config.device)

    def sample_time_step(self):
        t = torch.randint(low=1, high=self.MAX_STEPS, size=(self.batch_size, 1), device=self.config.device)
        return t

    def get_alpha(self, step):
        alpha_t = self.alpha_scheduler[step-1]
        return alpha_t
    
    def get_alpha_bar_t(self, t_tensor):
        alpha_bar_t = self.alpha_bar_scheduler[t_tensor-1].to(self.config.device)
        return alpha_bar_t

    def linear_beta_scheduler(self, timesteps):
        d = (0.02 - 10**(-4))/timesteps
        b_t = 1e-4 + torch.arange(1, timesteps+1) * d
        return b_t.to(self.config.device)

    def cosine_beta_scheduler(self, timesteps, beta_start=1e-4, beta_end=0.02):
        betas = torch.linspace(0, 1, timesteps)
        b_t = beta_start + 0.5 * (beta_end - beta_start) * (1 + torch.cos(math.pi * betas))
        return b_t.to(self.config.device)
