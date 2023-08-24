import json
from dataclasses import dataclass
from typing import List

import torch

from superphot_plus.constants import BATCH_SIZE, LEARNING_RATE


@dataclass
class NetworkParams:
    """Holds the neural network configuration."""

    input_dim: int
    output_dim: int
    neurons_per_layer: int
    num_hidden_layers: int

    def __iter__(self):
        return iter((self.input_dim, self.output_dim, self.neurons_per_layer, self.num_hidden_layers))


@dataclass
class ModelConfig:
    """Holds model training configuration."""

    network_params: NetworkParams

    normalization_means: List[float]
    normalization_stddevs: List[float]

    batch_size: int = BATCH_SIZE
    learning_rate: int = LEARNING_RATE

    device: torch.device = torch.device("cpu")

    def save(self, filename):
        """Save configuration data to a JSON file."""
        data_dict = {
            "network_params": [*self.network_params],
            "normalization_means": self.normalization_means,
            "normalization_stddevs": self.normalization_stddevs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_dict, f)

    @classmethod
    def load(cls, filename):
        """Load configuration data from a JSON file."""
        with open(filename, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        return ModelConfig(
            NetworkParams(*data_dict["network_params"]),
            data_dict["normalization_means"],
            data_dict["normalization_stddevs"],
            data_dict["batch_size"],
            data_dict["learning_rate"],
        )
