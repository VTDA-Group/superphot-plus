"""Data class for per-band curve priors"""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np


@dataclass
class PriorFields:
    """Holder for per-parameter field characterization"""

    clip_a: float = 0
    clip_b: float = 0
    mean: float = 0
    std: float = 0

    def to_numpy(self):
        """Fields as a length-4 numpy array."""
        return np.array([self.clip_a, self.clip_b, self.mean, self.std])


@dataclass
class CurvePriors:
    """Set of priors for fitting a single curve."""

    amp: PriorFields = None
    beta: PriorFields = None
    gamma: PriorFields = None
    t_0: PriorFields = None
    tau_rise: PriorFields = None
    tau_fall: PriorFields = None
    extra_sigma: PriorFields = None

    def __post_init__(self):
        """Additional logic to coerce string dictionaries into the appropriate
        data type."""
        if isinstance(self.amp, dict):
            self.amp = PriorFields(**self.amp)
            self.beta = PriorFields(**self.beta)
            self.gamma = PriorFields(**self.gamma)
            self.t_0 = PriorFields(**self.t_0)
            self.tau_rise = PriorFields(**self.tau_rise)
            self.tau_fall = PriorFields(**self.tau_fall)
            self.extra_sigma = PriorFields(**self.extra_sigma)

    def to_numpy(self):
        """Fields as a 7x4 numpy array"""
        return [
            self.amp.to_numpy(),
            self.beta.to_numpy(),
            self.gamma.to_numpy(),
            self.t_0.to_numpy(),
            self.tau_rise.to_numpy(),
            self.tau_fall.to_numpy(),
            self.extra_sigma.to_numpy(),
        ]


@dataclass
class MultibandPriors:
    """Set of per-band curve priors"""

    bands: Dict[str, CurvePriors] = field(default_factory=dict)
    """Per-band curve priors."""
    band_order: str = "ugrizy"
    """Ordering of bands."""
    reference_band: str = "r"
    """Reference band."""

    def __post_init__(self):
        """Additional logic to coerce string dictionaries into the appropriate
        data type."""
        for band, curve_priors in self.bands.items():
            if isinstance(curve_priors, dict):
                self.bands[band] = CurvePriors(**curve_priors)

    @property
    def ordered_bands(self):
        """Returns included bands in band_order."""
        bands_ordered = []
        for band in self.band_order:
            if band in self.bands:
                bands_ordered.append(band)

        return np.array(bands_ordered)

    @property
    def aux_bands(self):
        """Returns auxilliary (non-reference) bands in band_order."""
        bands_ordered = []
        for band in self.band_order:
            if band in self.bands and band != self.reference_band:
                bands_ordered.append(band)

        return np.array(bands_ordered)

    def filter_by_band(self, band_list, in_place=True):
        """Return MultibandPriors object with only some bands.
        """
        assert self.reference_band in band_list
        
        bands_filtered = {band: self.bands[band] for band in band_list}
        band_order = ""
        for band in self.band_order:
            if band in band_list:
                band_order += band
        
        if in_place:
            self.bands = bands_filtered
            self.band_order = band_order
            return self
        
        return MultibandPriors(
            bands=bands_filtered,
            band_order=band_order,
            reference_band=self.reference_band
        )        
        
    def to_numpy(self):
        """Fields as a (7*bands)x4 numpy array"""
        priors = []
        for band in self.band_order:
            if band in self.bands:
                priors.append(self.bands[band].to_numpy())

        return np.concatenate(priors)
