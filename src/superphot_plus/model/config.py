import dataclasses
from dataclasses import dataclass
from typing import List

import torch
import yaml
from typing_extensions import Self

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

    device = torch.device("cpu")

    def __post_init__(self):
        """Coerce string dictionaries into the appropriatedata type."""
        if isinstance(self.network_params, dict):
            self.network_params = NetworkParams(**self.network_params)  # pylint: disable=not-a-mapping

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
