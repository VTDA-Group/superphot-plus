"""Container for posterior samples from lightcurve fitting"""

import numpy as np

from superphot_plus.file_utils import get_posterior_filename, get_posterior_samples


class PosteriorSamples:
    """Container for posterior samples from lightcurve fitting"""

    def __init__(self, samples, name=None, sampling_method=None, sn_class=None):
        """A class for storing an manipulating posterior samples from a lightcurve.

        Parameters
        ----------
        samples : numpy array
            equal-weighted fitting samples for superphot model
        name : str, optional
            the name of the lightcurve
        sn_class : int, optional
            the classification of the supernova, if known.
        """
        self.samples = samples
        self.name = name
        self.sampling_method = sampling_method
        self.sn_class = sn_class

    def sample_mean(self):
        """Convenience method to get some summary statistics about the posterior samples.

        Returns
        -------
        sample_mean: np.array
            Mean of samples along the 0 axis.
        """
        return np.mean(self.samples, axis=0)

    def save_to_file(self, output_dir):
        """Save the posterior samples to a directory, using the lightcurve's name.

        Parameters
        ----------
        output_dir : str, optional
            Output directory path. Defaults to FITS_DIR.
        """
        posterior_filename = get_posterior_filename(self.name, output_dir, self.sampling_method)

        np.savez_compressed(posterior_filename, self.samples)

    @classmethod
    def from_file(cls, input_dir=None, name=None, sampling_method=None, sn_class=None):
        """Create a PosteriorSamples object from a file.

        Parameters
        ----------
        input_dir : str, optional
            Input directory path. Defaults to FITS_DIR.
        name : str
            Lightcurve name.
        sampling_method : str, optional
            Variety of sampler. Can be included in the sample file name.
        sn_class : int, optional
            the classification of the supernova, if known.
        """
        samples = get_posterior_samples(name, fits_dir=input_dir, sampler=sampling_method)

        return cls(samples, name=name, sampling_method=sampling_method, sn_class=sn_class)
