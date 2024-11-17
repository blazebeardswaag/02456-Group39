from dataclasses import dataclass

@dataclass
class Config:
    LR:float = 1e-4
    MAX_STEPS:int = 1000 
    DIM:tuple = (28, 28)
    MODEL_OUTPUT_PATH :str = "./model_serialzed"
