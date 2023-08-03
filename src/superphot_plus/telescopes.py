from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from typing_extensions import Self

from superphot_plus.priors.fitting_priors import MultibandPriors
from superphot_plus.utils import get_band_extinctions

@dataclass
class Telescope:
    """Holder for telescope-specific information."""

    name: str = ""
    priors: MultibandPriors = field(default_factory=MultibandPriors)
    wavelengths: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Check that priors and wavelengths are defined for all filters."""
        for b in self.priors.bands:
            assert b in self.wavelengths
            
    def get_ordered_wavelengths(self):
        """Return wavelengths in order that matches priors'
        filter order."""
        ordered_wvs = []
        for b in self.priors.ordered_bands:
            ordered_wvs.append(self.wavelengths[b])
        return np.array(ordered_wvs)
    
    def get_extinctions(self, ra, dec):
        """Get band extinctions at a specific coordinate.
        """
        ordered_b = self.priors.ordered_bands
        ordered_wvs = self.get_ordered_wavelengths()
        
        ext_list = get_band_extinctions(ra, dec, ordered_wvs)
        ext_dict = {ordered_b[i]: ext_list[i] for i in range(len(ext_list))}
        return ext_dict
    
    @classmethod
    def ZTF(cls) -> Self:
        """Get ZTF priors and wavelengths.
        """
        ztf_wvs = {"g": 4741.64, "r": 6173.23}
        return Telescope("ZTF", MultibandPriors.load_ztf_priors(), ztf_wvs)
        
        
            
