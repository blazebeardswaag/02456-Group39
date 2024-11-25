import json
from dataclasses import dataclass



@dataclass
class Experiment:
    experiment_name: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    device: str
    training_loss: list[float]
    validation_loss: list[float]
    training_time: float

    

class ExperimentTracker:
    
    def export_results(self, experiment: Experiment):
        experiment_name = experiment.experiment_name

        results_dict = {
            "experiment_name": experiment_name,
            "learning_rate": experiment.learning_rate,
            "batch_size": experiment.batch_size,
            "num_epochs": experiment.num_epochs,
            "device": experiment.device,
        }

        with open(f"{experiment_name}.json", "w") as f:
            json.dump(results_dict, f)



