from configs.config_manager import context_manager
import pprint
import torch.nn as nn 
import torch

with context_manager(

    experiment_name="mnist_training",
    use_wandb=True,
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
) as config:

# prints the sweep configs located in sweep_config file
pprint.pprint(config.sweep_config)