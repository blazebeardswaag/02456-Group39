
import torch 
from unet import ScoreNetwork0


def compute_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {total_params}")


model = ScoreNetwork0()     

compute_params(model)

 