from dataclasses import dataclass
import dataclasses
import os
import numpy as np
from typing_extensions import Self

import yaml


@dataclass
class SupernovaProperties:
    """Supernova physical parameters."""

    bfield: float = 0
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
    def check_property_exists(cls, parameter):
        """Checks if supernova property name exists.

        Parameters
        ----------
        parameter : str
            The name of the physical property.

        Returns
        -------
        bool
            True if property exists, false otherwise.
        """
        return parameter in [p.name for p in dataclasses.fields(SupernovaProperties)]

    @classmethod
    def get_property_by_name(cls, properties, name: str):
        """Returns the supernovae values for a specified property.

        Returns
        -------
        np.array
            The bundle of property values for a set of property objects.
        """
        return np.array([getattr(p, name) for p in properties])
