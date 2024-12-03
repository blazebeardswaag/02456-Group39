from configs.config_manager import context_manager
import pprint
import torch.nn as nn 
import torch

with context_manager(
    experiment_name="mnist_training",
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
) as config:

    print(config)
    
    print("-----------------"*5)
    print(config.sweep_config)

#sweep_id = wandb.sweep(config.sweep_config, project="default_project")

#andb.agent(sweep_id, train, count=5)