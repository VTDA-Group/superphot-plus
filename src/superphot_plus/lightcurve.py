"""A data class for storing information for a single light curve."""
import numpy as np

from superphot_plus.file_utils import read_single_lightcurve, save_single_lightcurve


class Lightcurve:
    def __init__(self, times, fluxes, flux_errors, bands, name=None, sn_class=None):
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
        sn_type : int, optional
            The classification of supernova (if known).

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
        self.sn_class = sn_class

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

    def find_max_flux(self, error_coeff=-1.0, band=None):
        """Find the value and timestamp of the maximum flux in
        a given band, accounting for the error estimate.

        Uses value[t] = fluxes[t] + error_coeff * | flux_errors[t] |

        Parameters
        ----------
        error_coeff : float
            The multiplicative constant to use when accounting for
            error. Default = -1.0 for flux - |error|.
        band : str, optional
            The band to check.  Use None to find the maximum over
            all bands.

        Returns
        -------
        value, timestamp : float, float
            Returns the maximum flux value and its corresponding time.

        Raises
        ------
        ValueError if the light curve is empty or there are no observations in the given band.
        """
        if band is None:
            ref_band = [True] * len(self.bands)
        else:
            ref_band = self.bands == band
        if np.all(ref_band == False) or len(self.bands) == 0:
            raise ValueError(f"ERROR: Light curve has no points in band {band}")

        adjusted_flux = self.fluxes[ref_band] + error_coeff * np.abs(self.flux_errors[ref_band])
        max_index = np.argmax(adjusted_flux)
        max_flux_time = self.times[ref_band][max_index]

        return adjusted_flux[max_index], max_flux_time

    def sort_by_time(self, in_place=True):
        """Sort the data by timestamp.

        Parameters
        ----------
        in_place : bool
            A Boolean indicating whether to modify the data in-place.

        Returns
        -------
        result : Lightcurve
            The padded lightcurve. Returns self if in_place == True.
        """
        sort_idx = np.argsort(self.times)
        if in_place:
            self.times = self.times[sort_idx]
            self.fluxes = self.fluxes[sort_idx]
            self.flux_errors = self.flux_errors[sort_idx]
            self.bands = self.bands[sort_idx]
            return self
        else:
            return Lightcurve(
                self.times[sort_idx],
                self.fluxes[sort_idx],
                self.flux_errors[sort_idx],
                self.bands[sort_idx],
                name=self.name,
            )

    def pad_bands(self, bands, size, in_place=True):
        """Truncate or pad the bands so that each band has
        exactly ``size`` entries.

        Parameters
        ----------
        bands : array-like
            An array of bands to include in the final data.
        size : int
            The required number of data points in each band.
        in_place : bool
            A Boolean indicating whether to modify the data in-place.

        Returns
        -------
        result : Lightcurve
            The padded lightcurve. Returns self if in_place == True.
        """
        lc = self.sort_by_time(in_place)

        t_padded = np.array([])
        f_padded = np.array([])
        ferr_padded = np.array([])
        b_padded = np.array([])

        for b in bands:
            matches = lc.bands == b
            len_b = len(lc.bands[matches])
            t_s = lc.times[matches]
            f_s = lc.fluxes[matches]
            ferr_s = lc.flux_errors[matches]
            b_s = lc.bands[matches]

            # If we have too many data points, use only the first ``size``
            # as ordered by time. Otherwise pad the data.
            if len_b > size:
                t_padded = np.append(t_padded, t_s[:size])
                f_padded = np.append(f_padded, f_s[:size])
                ferr_padded = np.append(ferr_padded, ferr_s[:size])
                b_padded = np.append(b_padded, b_s[:size])
            else:
                t_padded = np.append(t_padded, t_s)
                f_padded = np.append(f_padded, f_s)
                ferr_padded = np.append(ferr_padded, ferr_s)
                b_padded = np.append(b_padded, b_s)

                t_padded = np.append(t_padded, [5000] * (size - len_b))
                f_padded = np.append(f_padded, [0.0] * (size - len_b))
                ferr_padded = np.append(ferr_padded, [1e10] * (size - len_b))
                b_padded = np.append(b_padded, [b] * (size - len_b))

        # Depending on the setting in_place, lc is either self or a new copy
        # of the lightcurve.
        lc.times = t_padded
        lc.fluxes = f_padded
        lc.flux_errors = ferr_padded
        lc.bands = b_padded
        return lc

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
