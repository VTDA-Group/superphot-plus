from dataclasses import dataclass
import dataclasses
import os
from typing_extensions import Self

import yaml


@dataclass
class SupernovaProperties:
    """Supernova physical parameters."""

    bfield: int = 0
    pspin: float = 0
    mejecta: float = 0
    vejecta: float = 0

    def write_to_file(self, name: str):
        """Save configuration data to a YAML file."""
        args = dataclasses.asdict(self)
        encoded_string = yaml.dump(args, sort_keys=False)
        with open(name, "w", encoding="utf-8") as file_handle:
            file_handle.write(encoded_string)

    @classmethod
    def from_file(cls, input_dir, name: str) -> Self:
        """Load configuration data from a YAML file."""
        file_path = os.path.join(input_dir, name)
        with open(file_path, "r", encoding="utf-8") as file_handle:
            metadata = yaml.safe_load(file_handle)
            return cls(**metadata)

    @classmethod
    def filter_property(cls, properties, name: str):
        """Filters array of supernova properties by property name."""
        return [getattr(p, name) for p in properties]
