import json
from dataclasses import dataclass
from typing import List

import torch

from superphot_plus.constants import BATCH_SIZE, LEARNING_RATE


@dataclass
class ModelConfig:
    """Class that holds the MLP configuration."""

    input_dim: int
    output_dim: int
    neurons_per_layer: int
    num_hidden_layers: int

    normalization_means: List[float]
    normalization_stddevs: List[float]

    batch_size: int = BATCH_SIZE
    learning_rate: int = LEARNING_RATE

    device: torch.device = torch.device("cpu")

    def __iter__(self):
        return iter(
            (
                self.input_dim,
                self.output_dim,
                self.neurons_per_layer,
                self.num_hidden_layers,
                self.batch_size,
                self.learning_rate,
            )
        )

    def save(self, filename):
        """Save configuration data to a JSON file."""
        data_dict = {
            "config": [
                self.input_dim,
                self.output_dim,
                self.neurons_per_layer,
                self.num_hidden_layers,
                self.batch_size,
                self.learning_rate,
            ],
            "normalization_means": self.normalization_means,
            "normalization_stddevs": self.normalization_stddevs,
        }
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data_dict, f)

    @classmethod
    def load(cls, filename):
        """Load configuration data from a JSON file."""
        with open(filename, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
        return ModelConfig(
            *data_dict["config"],
            data_dict["normalization_means"],
            data_dict["normalization_stddevs"],
        )
