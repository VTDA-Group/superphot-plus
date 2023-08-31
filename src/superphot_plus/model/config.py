import dataclasses
from dataclasses import dataclass
from typing import List

import torch
import yaml
from ray import tune
from typing_extensions import Self

from superphot_plus.constants import BATCH_SIZE, LEARNING_RATE


@dataclass
class ModelConfig:
    """Holds model training configuration."""

    input_dim: int
    output_dim: int
    neurons_per_layer: int
    num_hidden_layers: int

    normalization_means: List[float]
    normalization_stddevs: List[float]

    batch_size: int = BATCH_SIZE
    learning_rate: int = LEARNING_RATE

    device = torch.device("cpu")

    def write_to_file(self, file: str):
        """Save configuration data to a YAML file."""
        args = dataclasses.asdict(self)
        encoded_string = yaml.dump(args, sort_keys=False)
        with open(file, "w", encoding="utf-8") as file_handle:
            file_handle.write(encoded_string)

    @classmethod
    def from_file(cls, file: str) -> Self:
        """Load configuration data from a YAML file."""
        with open(file, "r", encoding="utf-8") as file_handle:
            metadata = yaml.safe_load(file_handle)
            return cls(**metadata)


@dataclass
class TrainConfig:
    """Holds information about the specific training
    configuration of a model. The default values are
    sampled by ray tune for parameter optimization."""

    neurons_per_layer: int = tune.choice([128, 256, 512])
    num_hidden_layers: int = tune.choice([2, 3, 4])
    goal_per_class: int = tune.choice([100, 500, 1000])

    num_folds: int = tune.choice(list(range(5, 6)))
    num_epochs: int = tune.choice([250])

    batch_size: int = tune.choice([32, 64, 128])
    learning_rate: float = tune.loguniform(1e-4, 1e-1)
