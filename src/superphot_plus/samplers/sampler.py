"""Container for posterior samples for a single lightcurve"""
import abc
from typing import List

from superphot_plus.posterior_samples import PosteriorSamples


class Sampler(abc.ABC):
    """Base class for sampling algorithms."""

    @abc.abstractmethod
    def run_single_curve(self, lightcurve, priors, **kwargs) -> PosteriorSamples:
        """Docstring"""

    @abc.abstractmethod
    def run_multi_curve(self, lightcurves, priors, **kwargs) -> List[PosteriorSamples]:
        """Docstring"""
