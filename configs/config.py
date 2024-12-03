from dataclasses import dataclass, asdict
import json
import os
from datetime import datetime

@dataclass
class Config:
    LR:float = 1e-4
    MAX_STEPS:int = 1000 
    DIM:tuple = (28, 28)
    MODEL_OUTPUT_PATH :str = "./model_serialzed"

    # Wandb config
    use_wandb: bool = False
    
    # Training 
    batch_size: int = None
    num_epochs: int = None
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
    
    scheduler_type: str = "linear"
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
    
