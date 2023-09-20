"""A data class for storing information for a single light curve."""
import os

import numpy as np


class Lightcurve:
    """A class for storing and manipulating a light curve."""

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

    def _reindex(self, indices, in_place=True):
        """Rearrange or subset the values in each array to match
        the ordering provided by indices.

        Parameters
        ----------
        indices : array_like
            This can either be an array-like of integer indices to keep
            and the order in which to put them or an array-like of bools
            that indicate whether to keep each entry.
        in_place : bool
            A Boolean indicating whether to modify the data in-place or
            make a new copy of the light curve.

        Returns
        -------
        result : Lightcurve
            The resulting Lightcurve. Returns self if in_place == True.
        """
        if in_place:
            self.times = self.times[indices]
            self.fluxes = self.fluxes[indices]
            self.flux_errors = self.flux_errors[indices]
            self.bands = self.bands[indices]
            return self
        return Lightcurve(
            self.times[indices],
            self.fluxes[indices],
            self.flux_errors[indices],
            self.bands[indices],
            name=self.name,
            sn_class=self.sn_class,
        )

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
        return np.count_nonzero(self.bands == band)

    def unique_bands(self):
        """Return a list of unique bands in the data.

        Returns
        -------
        result : numpy array
            An array of the unique bands.
        """
        return np.unique(self.bands)

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
        if np.all(ref_band == False) or len(self.bands) == 0:  # pylint: disable=singleton-comparison
            raise ValueError(f"ERROR: Light curve has no points. band={band}")

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
        return self._reindex(sort_idx, in_place=in_place)

    def filter_by_band(self, keep_bands, in_place=True):
        """Filter the light curve to keep only the specified bands.

        Parameters
        ----------
        keep_bands : array-like
            An array of bands to include in the final data.
        in_place : bool
            A Boolean indicating whether to modify the data in-place.

        Returns
        -------
        result : Lightcurve
            The filtered lightcurve. Returns self if in_place == True.
        """
        keep_row = np.isin(self.bands, keep_bands)
        return self._reindex(keep_row, in_place=in_place)

    def band_as_int(self, ordered_bands, fail_on_missing=True):
        """Returns the band array as integers corresponding to the band's position in
        ordered_bands.

        Parameters
        ----------
        ordered_bands : array_like
            The order in which to enumerate the bands.
        fail_on_missing : bool
            Raise an exception if one of the bands is not included in the ordering.

        Returns
        -------
        result : array_like
            The list of bands as integers. Uses -1 for unspecified bands.

        Raises
        ------
        ValueError if a band in the light curve is not included in ordered_bands
        and fail_on_missing is True.
        """
        if not np.all(np.isin(self.bands, ordered_bands)) and fail_on_missing:
            raise ValueError("ERROR: Unmapped bands found in band_as_int.")

        result = np.array([-1] * len(self.bands))
        for idx, band in enumerate(ordered_bands):
            result[self.bands == band] = idx
        return result

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

        Raises
        ------
        FileExistsError if the file exists and overwrite is False.
        """
        if os.path.exists(filename) and not overwrite:  # pragma: no cover
            raise FileExistsError(f"ERROR: File already exists {filename}")

        lcs = np.array([self.times, self.fluxes, self.flux_errors, self.bands])
        np.savez_compressed(filename, lcs=lcs, name=self.name, sn_class=self.sn_class)

    def debug_string(self):
        """Create and return a human readable debugging string.

        Returns
        -------
        res : str
            The debugging string.
        """
        res = (
            f"Supernova (name={self.name}, class={self.sn_class}, size={len(self.times)})\n"
            f"  Times: {self.times}\n"
            f"  Fluxes: {self.fluxes}\n"
            f"  Flux Errors: {self.flux_errors}\n"
            f"  Bands: {self.bands}\n"
        )
        return res

    @classmethod
    def from_file(cls, filename, ref_band="r", t0_lim=None, shift_time=True):
        """Create a Lightcurve object from a file.

        Parameters
        ----------
        filename : str
            The name of the file to use.
        ref_band : str
            The reference band to use. Default = 'r'
        t0_lim : float, optional
            Upper limit for t0. Defaults to None.
        shift_time : bool
            Shift the time stamps so that the maximum flux in the reference
            band occurs at time = 0. Default = True.

        Returns
        -------
        A new Lightcurve object.

        Raises
        ------
        FileNotFoundError if the file does not exist.
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"ERROR: File does not exist {filename}")

        # Load the data as a numpy array.
        npy_array = np.load(filename, allow_pickle=True)
        arr = npy_array["lcs"]
        curve_name = npy_array["name"]
        sn_class = npy_array["sn_class"]

        # Keep only the rows without NaNs in the error column.
        good_rows = arr[2] != "nan"
        tdata = arr[0][good_rows].astype(float)
        fdata = arr[1][good_rows].astype(float)
        edata = arr[2][good_rows].astype(float)
        bdata = arr[3][good_rows]
        lc = Lightcurve(tdata, fdata, edata, bdata, name=curve_name, sn_class=sn_class)

        # Enforce the time ceiling if there is one.
        if t0_lim is not None:
            lc = lc._reindex(tdata <= t0_lim)

        # Shift the time to align 0.0 with maximum flux.
        if lc.obs_count(band=ref_band) > 0 and shift_time:
            _, max_flux_loc = lc.find_max_flux(band=ref_band)
            lc.times = lc.times - max_flux_loc

        return lc
