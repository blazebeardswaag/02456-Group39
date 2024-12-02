import torch
import math

def get_alpha(t):
    alpha_t = 1 - linear_beta_schedueler(t)
    return alpha_t

def linear_beta_schedueler(step):
    d = (0.02 - 10**(-4))/(1000) 
    b_t = 10**(-4) + step * d 
    return b_t

def cosine_beta_scheduler(timesteps, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(0, 1, timesteps)
    return beta_start + 0.5 * (beta_end - beta_start) * (1 + torch.cos(math.pi * betas))

def get_alpha_bar_t(t):
    alpha = 1.0 - get_alpha(t)  
    alpha_bar_t = torch.cumprod(alpha, dim=0)
    return alpha_bar_t 