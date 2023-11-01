import dataclasses
from dataclasses import dataclass
from typing import List, Optional

import torch
import yaml
from typing_extensions import Self


# pylint: disable=too-many-instance-attributes
@dataclass
class ModelConfig:
    """Holds information about the specific training
    configuration of a model. The default values are
    sampled by ray tune for parameter optimization."""

    input_dim: Optional[int] = None
    output_dim: Optional[int] = None

    normalization_means: Optional[List[float]] = None
    normalization_stddevs: Optional[List[float]] = None

    # Tunable parameters
    neurons_per_layer: Optional[int] = None
    num_hidden_layers: Optional[int] = None
    goal_per_class: Optional[int] = None
    num_folds: Optional[int] = None
    num_epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None

    best_val_loss: Optional[float] = None

    device = torch.device("cpu")

    def set_non_tunable_params(self, input_dim, output_dim, norm_means, norm_stddevs):
        """Adds information about the params that are not tunable."""
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.normalization_means = norm_means
        self.normalization_stddevs = norm_stddevs

    def set_best_val_loss(self, best_val_loss):
        """Sets the best validation loss from training."""
        self.best_val_loss = best_val_loss

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
