from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime

sweep_config = {
    'method': 'grid'
    }


metric = {
    'name': 'loss',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric

@dataclass
class Config:
    DIM:tuple = (28, 28)
    MODEL_OUTPUT_PATH :str = "./model_serialzed"

    # Wandb config
    use_wandb: bool = True
    
    # Parameters for sweep
    parameters_dict {
        'batch_size': {
            'values': [64,128,256,512,1024]
        }, 

        'scheduler_type':{
            'values':['linear', 'cosin']
        },

        'LR':{
            'values':[1e-3,1e-4,1e-5,1e-6]
        }

    }

    sweep_config['parameters'] = parameters_dict

    # DIctionary for values that won't be optimized.
    parameters_dict.update({
    'num_epochs': {
        'value': 2},
    'MAX_STEPS' {
        'value': 1000
    }
    
    })

    # Training 
    device: str = None
    training_losses: list = None
    
    # Metadata
    experiment_name: str = None
    timestamp: str = None
    
    # Early stopping configs
    patience: int = 5
    min_delta: float = 0.01
    monitor: str = "loss"
    mode: str = "min"
    
 
    def __post_init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_losses = []
        self.experiment_name = f"experiment_{self.timestamp}"
        
    def save(self):
        os.makedirs("experiments", exist_ok=True)
        filepath = os.path.join("experiments", f"{self.experiment_name}.json")
        
        config_dict = asdict(self)
        config_dict['DIM'] = list(self.DIM) 
        
        if hasattr(self, 'device') and hasattr(self.device, 'type'):
            config_dict['device'] = str(self.device)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
