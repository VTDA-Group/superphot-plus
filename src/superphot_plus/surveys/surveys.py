"""Data class for survey-specific configuration parameters."""

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import yaml
from typing_extensions import Self

import superphot_plus
from superphot_plus.surveys.fitting_priors import MultibandPriors
from superphot_plus.utils import get_band_extinctions


@dataclass
class Survey:
    """Holder for survey-specific information."""

    name: str = ""
    priors: MultibandPriors = field(default_factory=MultibandPriors)
    wavelengths: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Check that priors and wavelengths are defined for all filters.

        Perform additional logic to coerce string dictionaries into the appropriate
        data type.
        """
        if isinstance(self.priors, dict):
            self.priors = MultibandPriors(**self.priors)  # pylint: disable=not-a-mapping
        for band in self.priors.bands:
            assert band in self.wavelengths

    def get_ordered_wavelengths(self):
        """Return wavelengths in order that matches priors'
        filter order.

        Returns
        ----------
        ordered_wvs : np.ndarray
            Bands' wavelengths in order matching priors.
        """
        ordered_wvs = []
        for band in self.priors.ordered_bands:
            ordered_wvs.append(self.wavelengths[band])
        return np.array(ordered_wvs)

    def get_extinctions(self, ra, dec, mwebv=None):
        """Get band extinctions at a specific coordinate.

        Parameters
        ----------
        ra : float
            The right ascension of the object of interest, in degrees.
        dec : float
            The declination of the object of interest, in degrees.
        mwebv : float
            Milky Way extinction value. If given, use this directly.

        Returns
        ----------
        ext_dict : dict
            Maps bands to extinction magnitudes.
        """
        ordered_b = self.priors.ordered_bands
        ordered_wvs = self.get_ordered_wavelengths()
        
        if mwebv is not None:
            ext_list = get_band_extinctions_from_mwebv(mwebv, ordered_wvs)
        else:
            ext_list = get_band_extinctions(ra, dec, ordered_wvs)
            
        ext_dict = {ordered_b[i]: ext_list[i] for i in range(len(ext_list))}
        return ext_dict

    def write_to_file(self, file: str):
        """Write per-band curve priors to a yaml file."""
        args = dataclasses.asdict(self)
        encoded_string = yaml.dump(args, sort_keys=False)
        with open(file, "w", encoding="utf-8") as file_handle:
            file_handle.write(encoded_string)

    @classmethod
    def from_file(cls, file: str) -> Self:
        """Read per-band curve priors from a yaml file."""
        with open(file, "r", encoding="utf-8") as file_handle:
            metadata = yaml.safe_load(file_handle)
            return cls(**metadata)

    @classmethod
    def ZTF(cls) -> Self:  # pylint: disable=invalid-name
        """Get ZTF priors and wavelengths.

        Returns
        ----------
        Survey
            Survey object representing the Zwicky Transient Facility (ZTF).
        """
        package_filepath = os.path.dirname(superphot_plus.__file__)
        yaml_file = os.path.join(package_filepath, "surveys", "ztf.yaml")
        return cls.from_file(yaml_file)
    
    
    @classmethod
    def LSST(cls) -> Self:  # pylint: disable=invalid-name
        """Get LSST priors and wavelengths.

        Returns
        ----------
        Survey
            Survey object representing the Rubin Observatory's LSST.
        """
        package_filepath = os.path.dirname(superphot_plus.__file__)
        yaml_file = os.path.join(package_filepath, "surveys", "lsst.yaml")
        return cls.from_file(yaml_file)
