"""A data class for storing information for a single light curve."""
import numpy as np

from superphot_plus.file_utils import read_single_lightcurve, save_single_lightcurve


class Lightcurve:
    def __init__(self, times, fluxes, flux_errors, bands, name=None):
        """A class for storing and manipulating a light curve.

        Parameters
        ----------
        times : numpy array
            The time stamps of the light curve data
        fluxes : numpy array
            The light curve fluxes
        flux_errors : numpy array
            The light curve flux errors
        bands : numpy array
            The band labels
        name : str, optional
            The name of the light curve.

        Raises
        ------
        ValueError if the arrays are of different lengths.
        """
        num_pts = len(times)
        if len(fluxes) != num_pts or len(flux_errors) != num_pts or len(bands) != num_pts:
            raise ValueError("Lightcurve: All arrays must be equal size.")
        self.times = times
        self.fluxes = fluxes
        self.flux_errors = flux_errors
        self.bands = bands
        self.name = name

    def obs_count(self, band=None):
        """Return the count of observations (in a given band).

        Parameters
        ----------
        band : str, optional
            The band to count.  Use None to count all bands.

        Returns
        -------
        count : int
            The observation count
        """
        if band is None:
            return len(self.times)
        else:
            return np.count_nonzero(self.bands == band)

    def save_to_file(self, filename, overwrite=False):
        """Write the light curve to a file.

        Parameters
        ----------
        filename : str
            The file name to use.
        overwrite : bool, optional
            Overwrite existing files.
        """
        save_single_lightcurve(
            filename, self.times, self.fluxes, self.flux_errors, self.bands, overwrite=overwrite
        )

    @classmethod
    def from_file(cls, filename, t0_lim=None):
        """Create a Lightcurve object from a file.

        Parameters
        ----------
        filename : str
            The name of the file to use.
        t0_lim : float, optional
            Upper limit for t0. Defaults to None.

        Returns
        -------
        A new Lightcurve object.
        """
        print(f"Loading light curve from {filename}")
        tdata, fdata, ferrdata, bdata = read_single_lightcurve(filename, t0_lim)
        curve_name = filename.split("/")[-1][:-4]
        return Lightcurve(tdata, fdata, ferrdata, bdata, name=curve_name)
