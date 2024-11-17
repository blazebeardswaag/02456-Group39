import torch 

def sample_epsilon(self, xT):
    eps = torch.normal(mean=0.0, std=1.0, size=xT)
    return eps    


def get_alpha(t):
    alpha_t = 1 - linear_beta_schedueler(t)
    return alpha_t


def linear_beta_schedueler(step):
    d = (0.02 - 10**(-4))/(1000) 
    b_t = 10**(-4) + step * d 
    return b_t


def get_alpha_bar_t(self, t):

    alpha = 1.0 - get_alpha(t)  
    alpha_bar_t = torch.cumprod(alpha, dim=0)  # Cumulative product
   # alpha_bar_t = alpha_bar_t.view(self.batch_size, 1, 1, 1)

    return alpha_bar_t



