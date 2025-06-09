import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

from snapi import LightCurve, Filter, Photometry
from superphot_plus.utils import flux_model, params_valid
from superphot_plus.priors import generate_priors


def ztf_noise_model(mag, snr_range=None):
    """A very, very simple noise model which assumes the dimmest magnitude is at SNR = 1,
    and the brightest mag is at SNR = 10.

    Parameters
    ----------
    mag : np.ndarray
        Observed magnitudes.
    band : np.ndarray
        Observed bands (g or r).
    snr_range_g : tuple
        Range of signal-to-noise ratios desired in g-band. Defaults to [1, 10]
    snr_range_r : tuple
        Range of signal-to-noise ratios desired in r-band. Defaults to [1, 10]

    Returns
    ----------
    snr : np.ndarray
        Signal-to-noise ratios (SNR) of the observations.
    """
    if not snr_range:
        snr_range = [1, 10]

    mag_range = np.max(mag) - np.min(mag)

    snr = (snr_range[1] - snr_range[0]) * (
        mag - np.min(mag)
    ) / mag_range + snr_range[0]

    return snr


def create_clean_models(nmodels, num_times=100):
    """Generate 'clean' (noiseless) models from the prior

    Parameters
    ----------
    nmodels : int
        The number of models you want to generate.
    num_times : int, optional
        The number of timesteps to use. Default = 100
    bands : list, optional
        The ordered list of bands to use. Default = ['r', 'g']
    ref_band : str, optional
        The reference band. Default = 'r'

    Returns
    -------
    params : array-like of numpy arrays
        The array of parameters used to generate each model.
    lcs : array-like of numpy arrays
        The array of individual light curves for each model generated.
    """
    params = []    
    bands = ["ZTF_r", "ZTF_g"]
    
    tdata = np.linspace(-50, 150, num_times)
    edata = np.asarray([1e-6] * num_times, dtype=float)

    priors = generate_priors(["ZTF_r", "ZTF_g"])
    params = []
    all_phots = []

    while len(all_phots) < nmodels:
        valid = True
        orig_cube = priors.sample(cube=None)
        lcs = []

        for i, b in enumerate(bands):
            band_mask = [b in param for param in priors.dataframe.param]
            cube = orig_cube[band_mask][:,np.newaxis]

            # Try again if we picked invalid priors.
            if not params_valid(cube):
                valid = False
                break
                
            f_model = flux_model(cube, tdata, None)[0]
            lc = LightCurve.from_arrays(
                phase=tdata,
                flux=f_model,
                flux_unc=edata,
                filt=Filter(
                    instrument='ZTF',
                    band=b.split("_")[1],
                    center=np.nan
                )
            )
            lcs.append(lc)
        if valid:
            params.append(orig_cube)
            phot = Photometry.from_light_curves(lcs, phased=True)
            all_phots.append(phot)

    return params, all_phots


def create_ztf_model(plot=False):
    """Generate realisitic-ish ZTF light curves from the Superphot+ prior.

    Parameters
    ----------
    plot : bool
        Whether resulting light curve is plotted and saved. Defaults to False.

    Returns
    ----------
    params : np.ndarray
        Set of parameters used to generate model.
    tdata : np.ndarray
        Time values of each datapoint.
    filter_data : np.ndarray
        Filter corresponding to each datapoint.
    dirty_model : np.ndarray
        Dirty flux values at each time value.
    sigmas : np.ndarray
        Uncertainties of each dirty flux value.
    """
    # This is going to random simulate some observation every 2-3 days across 2 filters
    params = []    
    bands = ["ZTF_r", "ZTF_g"]
    
    num_times = 100
    tdata = np.linspace(-50, 150, num_times)

    priors = generate_priors(["ZTF_r", "ZTF_g"])
    phot = None
    orig_cube = None
    valid = False

    while not valid:
        valid = True
        orig_cube = priors.sample(cube=None)
        lcs = []

        for _, b in enumerate(bands):
            band_mask = [b in param for param in priors.dataframe.param]
            cube = orig_cube[band_mask][:, np.newaxis]

            # Try again if we picked invalid priors.
            if not params_valid(cube):
                valid = False
                break
                
            f_model = flux_model(cube, tdata, None)[0]
            snr = ztf_noise_model(f_model)
            gind = np.where(snr > 3)  # any points with SNR < 3 are ignored
            snr = snr[gind]
            f_model = f_model[gind]
            tdata = tdata[gind]
            sigmas = f_model / snr
            dirty_model = f_model + np.random.normal(0, sigmas)

            lc = LightCurve.from_arrays(
                phase=tdata,
                flux=dirty_model,
                flux_unc=sigmas,
                filt=Filter(
                    instrument='ZTF',
                    band=b.split("_")[1],
                    center=np.nan
                )
            )
            lcs.append(lc)
            
        if valid:
            phot = Photometry.from_light_curves(lcs, phased=True)

    if plot:
        _, ax = plt.subplots()
        phot.plot(ax=ax)
        ax.set_xlabel("Time (days)")
        ax.set_ylabel("Flux (arbitrary units)")
        plt.show()

    return orig_cube, phot


# Can run this with create_model(plot=True)
if __name__ == "__main__":
    create_ztf_model(plot=True)