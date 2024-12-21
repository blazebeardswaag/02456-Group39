import torch

def sample_epsilon(size):
    """Sample random noise."""
    return torch.randn(size)

def get_alpha(t):
    """Calculate alpha value for timestep t."""
    return 1 - linear_beta_schedueler(t)

def linear_beta_schedueler(t):
    """Linear schedule for beta values."""
    beta_start = 0.0001
    beta_end = 0.02
    return beta_start + t * (beta_end - beta_start)

def cosine_beta_scheduler(t):
    """Cosine schedule for beta values."""
    s = 0.008
    max_beta = 0.999
    t = t.float()
    return torch.clamp((1 + t/1000) / (1 + s) * max_beta, 0, 0.999)

def get_alpha_bar_t(t):
    """Calculate cumulative product of alphas up to time t."""
    alpha = get_alpha(t)
    alpha_bar = torch.prod(alpha)
    return alpha_bar 