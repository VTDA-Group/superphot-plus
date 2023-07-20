import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import truncnorm

# Nested sampling priors
PRIOR_A = [-0.2, 1.2, 0.0, 0.5]
PRIOR_BETA = [0.0, 0.02, 0.0052, 1.5 * 0.000336]
PRIOR_GAMMA = [-2.0, 2.5, 1.1391, 1.5 * 0.1719]
PRIOR_T0 = [-100.0, 200.0, 0.0, 50.0]
PRIOR_TAU_RISE = [-1.0, 3.0, 0.5990, 1.5 * 0.2073]
PRIOR_TAU_FALL = [0.5, 4.0, 1.4296, 1.5 * 0.1003]
PRIOR_EXTRA_SIGMA = [-5.0, -0.5, -1.5364, 0.2691]

PRIOR_A_g = [0.0, 5.0, 1.0607, 1.5 * 0.1544]
PRIOR_BETA_g = [1.0, 1.07, 1.0424, 0.0026]
PRIOR_GAMMA_g = [0.8, 1.2, 1.0075, 0.0139]
PRIOR_T0_g = [1.0 - 0.0006, 1.0006, 0.9999 + 8.9289e-5, 1.5 * 4.5055e-05]
PRIOR_TAU_RISE_g = [0.5, 2.0, 0.9663, 0.0128]
PRIOR_TAU_FALL_g = [0.1, 3.0, 0.5488, 0.0553]
PRIOR_EXTRA_SIGMA_g = [0.2, 2.0, 0.8606, 0.0388]

max_flux = 1  # This is just to get it working


def trunc_gauss(quantile, clip_a, clip_b, mean, std):
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.ppf(quantile, a, b, loc=mean, scale=std)


def params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
    """
    Checks if params are valid given certain model constraints.
    """
    if tau_fall > 1.0 / beta:
        return False

    if gamma > (1.0 - beta * tau_fall) / beta:
        return False

    if tau_rise * (1.0 + np.exp(gamma / tau_rise)) < tau_fall:
        return False

    return True


def flux_model(cube, t_data, b_data):
    """
    This function is used utils.py!
    """
    A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7]  # pylint: disable=unused-variable

    if not params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
        return 1e10 * np.ones(len(t_data))

    phase = t_data - t0
    f_model = (
        A / (1.0 + np.exp(-phase / tau_rise)) * (1.0 - beta * gamma) * np.exp((gamma - phase) / tau_fall)
    )
    f_model[phase < gamma] = (
        A / (1.0 + np.exp(-phase[phase < gamma] / tau_rise)) * (1.0 - beta * phase[phase < gamma])
    )

    # for secondary band
    start_idx = 7
    A_b = A * cube[start_idx]
    beta_b = beta * cube[start_idx + 1]
    gamma_b = gamma * cube[start_idx + 2]
    t0_b = t0 * cube[start_idx + 3]
    tau_rise_b = tau_rise * cube[start_idx + 4]
    tau_fall_b = tau_fall * cube[start_idx + 5]

    inc_band_ix = np.array(b_data) == "g"
    phase_b = (t_data - t0_b)[inc_band_ix]
    phase_b2 = (t_data - t0_b)[inc_band_ix & (t_data - t0_b < gamma_b)]

    f_model[inc_band_ix] = (
        A_b
        / (1.0 + np.exp(-phase_b / tau_rise_b))
        * (1.0 - beta_b * gamma_b)
        * np.exp((gamma_b - phase_b) / tau_fall_b)
    )
    f_model[inc_band_ix & (t_data - t0_b < gamma_b)] = (
        A_b / (1.0 + np.exp(-phase_b2 / tau_rise_b)) * (1.0 - phase_b2 * beta_b)
    )
    return f_model


def create_prior(cube, tdata):
    """
    Creates prior for pymultinest, where each side
    of the "cube" is a value sampled between 0 and 1
    representing each parameter.

    This function is used in ztf_transient_fit.py!
    """

    cube[0] = max_flux * 10 ** (
        trunc_gauss(cube[0], *PRIOR_A)
    )  # log-uniform for A from 1.0x to 16x of max flux
    cube[1] = trunc_gauss(cube[1], *PRIOR_BETA)  # beta UPDATED, looks more Lorentzian so widened by 1.5x
    cube[2] = 10 ** trunc_gauss(cube[2], *PRIOR_GAMMA)  # very broad Gaussian temporary solution for gamma
    max_flux_loc = np.median(tdata)  # THIS IS DISTINCT FROM OTHER FUNC
    cube[3] = trunc_gauss(cube[3], np.amin(tdata) - 50.0, np.amax(tdata) + 50.0, max_flux_loc, 20.0)  # t0
    cube[4] = 10 ** (trunc_gauss(cube[4], *PRIOR_TAU_RISE))  # taurise, UPDATED
    cube[5] = 10 ** (trunc_gauss(cube[5], *PRIOR_TAU_FALL))  # tau fall UPDATED
    cube[6] = 10 ** (trunc_gauss(cube[6], *PRIOR_EXTRA_SIGMA))  # lognormal for extrasigma, UPDATED

    # green band
    cube[7] = trunc_gauss(cube[7], *PRIOR_A_g)  # A UPDATED
    cube[8] = trunc_gauss(cube[8], *PRIOR_BETA_g)  # beta UPDATED
    cube[9] = trunc_gauss(cube[9], *PRIOR_GAMMA_g)  # gamma, GAUSSIAN not Lorentzian
    cube[10] = trunc_gauss(cube[10], *PRIOR_T0_g)  # t0 UPDATED
    cube[11] = trunc_gauss(cube[11], *PRIOR_TAU_RISE_g)  # taurise UPDATED, Gaussian
    cube[12] = trunc_gauss(cube[12], *PRIOR_TAU_FALL_g)  # taufall UPDATED
    cube[13] = trunc_gauss(cube[13], *PRIOR_EXTRA_SIGMA_g)  # extra sigma UPDATED, Gaussian

    return cube

def ztf_noise_model(mag, band, snr_range_g = [1,10], snr_range_r=[1,10]):
    """
    A very, very simple noise model which assumes the dimmest magnitude is at SNR = 1, and the brightest mad is at SNR = 10

    inputs: 
    mag: observed magnitude
    band: observed band (g or r)
    snr_range_g: Range of signal-to-noise ratios you want to see in g-band
    snr_range_r: Range of signal-to-noise ratios you want to see in g-band

    output:
    snr: SNR of the observation.
    """

    snr = mag * 0 # set up a dummy array for the snr
    gind_g = np.where(band == 'g') # let's do g-band first
    snr[gind_g] = (snr_range_g[1] - snr_range_g[0]) / (np.max(mag[gind_g]) - np.min(mag[gind_g])) * (mag[gind_g] - np.min(mag[gind_g])) + snr_range_g[0]
    gind_r = np.where(band == 'r') # let's do g-band first
    snr[gind_r] = (snr_range_r[1] - snr_range_r[0]) / (np.max(mag[gind_r]) - np.min(mag[gind_r])) * (mag[gind_r]- np.min(mag[gind_r])) + snr_range_r[0]
    return snr
    



def create_model(plot=False):
    """
    Generate realisitic-ish ZTF light curves from the superphot+ prior

    inputs:
    plot: Boolean to just turn on some plotting functionality

    outputs:
    params : set of parameters used to generate model
    tdata : time array for light curve
    filter_data : filter array for light curve
    dirty_model : flux values
    sigmas = : errors on flux

    """

    # Randomly sample from parameter space and from data/filters
    cube = np.random.uniform(0, 1, 14)

    # This is going to random simulate some observation every 2-3 days across 2 filters
    num_observations = 130
    tdata = np.random.uniform(-100, 100, num_observations)
    filter_data = np.random.choice(['g','r'],size=num_observations)

    cube = create_prior(cube, tdata)
    A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7]  # pylint: disable=unused-variable

    # THIS IS A BAD SOLUTION -- it just skips if there is no good model, but it should actually attempt to regenerate until it gets a "good" model
    if not params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
        return('Failure')
    f_model = flux_model(cube, tdata, filter_data)
    snr = ztf_noise_model(f_model, filter_data)


    gind = np.where(snr>3)
    snr = snr[gind]
    f_model = f_model[gind]
    tdata = tdata[gind]
    filter_data = filter_data[gind]
    sigmas = f_model/snr

    dirty_model = f_model + np.random.normal(0,sigmas)

    if plot:
        plt.scatter(tdata, dirty_model, color=filter_data)
        plt.errorbar(tdata, dirty_model, yerr=sigmas, color='grey', alpha=0.2, linestyle='None')
        plt.xlabel('Time (days)')
        plt.ylabel('Flux (arbitrary units)')
        plt.show()
    return cube[:,7], tdata, filter_data, dirty_model, sigmas

# Can run this with create_model(plot=True)
