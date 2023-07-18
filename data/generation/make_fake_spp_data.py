import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Nested sampling priors
PRIOR_A = [-0.2, 1.2, 0., 0.5]
PRIOR_BETA = [0., 0.02, 0.0052, 1.5 * 0.000336]
PRIOR_GAMMA = [-2., 2.5, 1.1391, 1.5 * .1719]
PRIOR_T0 = [-100., 200., 0., 50.]
PRIOR_TAU_RISE = [-1.0, 3., 0.5990, 1.5 * 0.2073]
PRIOR_TAU_FALL = [0.5, 4., 1.4296, 1.5 * 0.1003]
PRIOR_EXTRA_SIGMA = [-5., -0.5, -1.5364, 0.2691]

PRIOR_A_g = [0., 5., 1.0607, 1.5 * 0.1544]
PRIOR_BETA_g = [1., 1.07, 1.0424, 0.0026]
PRIOR_GAMMA_g = [0.8, 1.2, 1.0075, 0.0139]
PRIOR_T0_g = [1. - 0.0006, 1.0006, 0.9999 + 8.9289e-5, 1.5 * 4.5055e-05]
PRIOR_TAU_RISE_g = [0.5, 2., 0.9663, 0.0128]
PRIOR_TAU_FALL_g = [0.1, 3., 0.5488, 0.0553]
PRIOR_EXTRA_SIGMA_g = [0.2, 2., 0.8606, 0.0388]

max_flux = 1 # This is just to get it working

def trunc_gauss(quantile, clip_a, clip_b, mean, std):
    a, b = (clip_a - mean) / std, (clip_b - mean) / std
    return truncnorm.ppf(quantile, a, b, loc=mean, scale=std)


def params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
    """
    Checks if params are valid given certain model constraints.
    """
    if tau_fall > 1. / beta:
        return False

    if gamma > (1. - beta * tau_fall) / beta:
        return False

    if tau_rise * (1. + np.exp(gamma / tau_rise)) < tau_fall:
        return False

    return True

def flux_model(cube, t_data, b_data):
    '''
    This function is used utils.py!
    '''
    A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7] # pylint: disable=unused-variable

    if not params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
        return 1e10 * np.ones(len(t_data))

    phase = t_data - t0
    f_model = (
        A
        / (1.0 + np.exp(-phase / tau_rise))
        * (1.0 - beta * gamma)
        * np.exp((gamma - phase) / tau_fall)
    )
    f_model[phase < gamma] = (
        A
        / (1.0 + np.exp(-phase[phase < gamma] / tau_rise))
        * (1.0 - beta * phase[phase < gamma])
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
    cube[1] = trunc_gauss(
        cube[1], *PRIOR_BETA
    )  # beta UPDATED, looks more Lorentzian so widened by 1.5x
    cube[2] = 10 ** trunc_gauss(
        cube[2], *PRIOR_GAMMA
    )  # very broad Gaussian temporary solution for gamma
    max_flux_loc = np.median(tdata) # THIS IS DISTINCT FROM OTHER FUNC
    cube[3] = trunc_gauss(
        cube[3], np.amin(tdata) - 50.0, np.amax(tdata) + 50.0, max_flux_loc, 20.0
    )  # t0
    cube[4] = 10 ** (trunc_gauss(cube[4], *PRIOR_TAU_RISE))  # taurise, UPDATED
    cube[5] = 10 ** (trunc_gauss(cube[5], *PRIOR_TAU_FALL))  # tau fall UPDATED
    cube[6] = 10 ** (
        trunc_gauss(cube[6], *PRIOR_EXTRA_SIGMA)
    )  # lognormal for extrasigma, UPDATED


    # green band
    cube[7] = trunc_gauss(cube[7], *PRIOR_A_g)  # A UPDATED
    cube[8] = trunc_gauss(cube[8], *PRIOR_BETA_g)  # beta UPDATED
    cube[9] = trunc_gauss(cube[9], *PRIOR_GAMMA_g)  # gamma, GAUSSIAN not Lorentzian
    cube[10] = trunc_gauss(cube[10], *PRIOR_T0_g)  # t0 UPDATED
    cube[11] = trunc_gauss(cube[11], *PRIOR_TAU_RISE_g)  # taurise UPDATED, Gaussian
    cube[12] = trunc_gauss(cube[12], *PRIOR_TAU_FALL_g)  # taufall UPDATED
    cube[13] = trunc_gauss(
        cube[13], *PRIOR_EXTRA_SIGMA_g
    )  # extra sigma UPDATED, Gaussian

    return cube

def create_clean_models(nmodels):
    '''
    Generate "clean" (noiseless) models from the prior

    inputs:
    nmodels : number of models you want to generate

    outputs: 
    params : set of parameters used to generate each model
    lcs : light curves for each model generated
    '''
    for i in np.arange(nmodels):
        cube = np.random.uniform(0, 1, 14)
        tdata = np.linspace(-100,100,100)
        cube = create_prior(cube, tdata)
        A, beta, gamma, t0, tau_rise, tau_fall, es = cube[:7] # pylint: disable=unused-variable

        if not params_valid(A, beta, gamma, t0, tau_rise, tau_fall):
            continue
        print(cube)
        bdata = np.asarray(['g']*100, dtype=str)
        f_model = flux_model(cube, tdata, bdata)
        plt.plot(tdata, f_model,'.')

# ASHLEY is still working on this...
create_clean_models(100)
plt.show()

