import torch 
import torch.nn as nn
from .image_generator import ImageGenerator

class Sampler:
    def __init__(self, config, batch_size):
        self.config = config 
        self.dim = self.config.DIM
        self.batch_size = batch_size

        
    def sample_time_step(self):
        t = torch.randint(low=1, high=self.config.MAX_STEPS, size=(self.batch_size,1))
        return t 
    
    def generate_noisy_image(self,t, image):
        noisy_image, generated_noise = self.sampler.add_noise(image, t)
        return noisy_image, generated_noise

    def sample_epsilon(self, xT):
        eps = torch.normal(mean=0.0, std=1.0, size=xT)
        return eps    
    
    def sample_gaussian(self, xT):
        eps = torch.normal(mean=0.0, std=1.0, size=xT)
        return eps    

    def get_alpha(self, t):
        beta_t = self.linear_beta_schedueler(t)
        alpha_t = 1 - beta_t
        return alpha_t

    def get_alpha_bar_t(self, t_tensor):

        batch_size = t_tensor.shape[0]
        alpha_bar_results = []
        for i in range(batch_size):
            t = t_tensor[i].item()
            alphas = torch.tensor([self.get_alpha(t_i) for t_i in range(1, t + 1)], dtype=torch.float32)
            alpha_bar_t = alphas.prod()
            alpha_bar_results.append(alpha_bar_t)
        return torch.stack(alpha_bar_results)

    def linear_beta_schedueler(self, step):
        d = (0.02 - 10**(-4))/(1000) 
        b_t = 10**(-4) + step * d 
        return b_t

class NoiseSchedueler: 
    def __init__(self):
        pass 

    def linear_beta_schedueler(self, step):
        d = (0.02 - 10**(-4))/(1000) 
        b_t = 10**(-4) + step * d 
        return b_t
