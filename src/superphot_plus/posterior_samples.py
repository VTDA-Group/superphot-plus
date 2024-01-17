"""Container for posterior samples from lightcurve fitting"""

import numpy as np

from superphot_plus.file_utils import get_posterior_filename, get_posterior_samples


class PosteriorSamples:
    """Container for posterior samples from lightcurve fitting"""

    def __init__(
        self, samples,
        name=None, sampling_method=None,
        sn_class=None, sample_mean=None,
        max_flux=None, redshift=None,
        **kwargs
    ):
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
        self.samples = np.atleast_2d(samples)
        self.name = name
        self.sampling_method = sampling_method
        self.sn_class = sn_class
        self.max_flux = max_flux
        self.redshift = redshift
        
        self._sample_mean = sample_mean
        
        if self._sample_mean is None:
            self._sample_mean = np.mean(self.samples, axis=0)

    def sample_mean(self):
        """Convenience method to get some summary statistics about the posterior samples.

        Returns
        -------
        sample_mean: np.array
            Mean of samples along the 0 axis.
        """
        return self._sample_mean

    def save_to_file(self, output_dir):
        """Save the posterior samples to a directory, using the lightcurve's name.

        Parameters
        ----------
        output_dir : str, optional
            Output directory path. Defaults to FITS_DIR.
        """
        posterior_filename = get_posterior_filename(self.name, output_dir, self.sampling_method)
        
        np.savez_compressed(
            posterior_filename,
            samples=self.samples,
            name=self.name,
            sampling_method=self.sampling_method,
            sn_class=self.sn_class,
            redshift=self.redshift,
            max_flux=self.max_flux
        )

    @classmethod
    def from_file(cls, input_dir, name, sampling_method=None, **kwargs):
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
        samples, all_kwargs = get_posterior_samples(name, fits_dir=input_dir, sampler=sampling_method)

        for k in kwargs:
            all_kwargs[k] = kwargs[k]
            
        if 'name' not in all_kwargs:
            all_kwargs['name'] = name
        if (sampling_method is not None) and ('sampling_method' not in all_kwargs):
            all_kwargs['sampling_method'] = sampling_method
            
        return cls(samples, **all_kwargs)
