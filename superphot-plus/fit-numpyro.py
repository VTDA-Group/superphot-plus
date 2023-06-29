import numpy as np
import jax.numpy as jnp
import matplotlib
#matplotlib.use('AGG') 
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import astropy.constants as c
import astropy.units as u
from scipy import interpolate
import numpyro
from numpyro.diagnostics import hpdi
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
from jax import random, vmap
from jax.config import config
config.update("jax_enable_x64", True)
import arviz as az
from astropy.cosmology import WMAP9 as cosmo
from numpyro.infer import Predictive
import extinction
import os

from constants import *
from file_paths import *
from utils import *



def import_data(fn, t0_lim=None):
    """
    Import the datafile.
    """
    npy_array = np.load(fn)
    arr = npy_array['arr_0']
    
    ferr = arr[2]
    t = arr[0][ferr != "nan"].astype(float)
    f = arr[1][ferr != "nan"].astype(float)
    b = arr[3][ferr != "nan"]
    ferr = ferr[ferr != "nan"].astype(float)

    if t0_lim is not None:
        f = f[t <= t0_lim]
        b = b[t <= t0_lim]
        ferr = ferr[t <= t0_lim]
        t = t[t <= t0_lim]
        
    return t, f, ferr, b

def trunc_norm(low, high, loc, scale):
    """
    Helper function for dist.TruncatedNormal()
    """
    return dist.TruncatedNormal(loc=loc, scale=scale, low=low, high=high)

def run_mcmc(fn, t0_lim=None):
    """
    Run dynesty importance nested sampling on datafile. Returns
    set of equally weighted posteriors (sets of fit parameters).
    """
    rng_key = random.PRNGKey(4)
    rng_key, rng_key_ = random.split(rng_key)

    ref_band_idx = 1 # red band
    
    prefix = fn.split("/")[-1][:-4]
    
    print(prefix)
    n_params = 14

    tdata, fdata, ferrdata, bdata = import_data(fn, t0_lim)
    
    if (tdata[bdata == "r"] is None) or (len(tdata[bdata == "r"]) == 0):
        return None
    if (tdata[bdata == "g"] is None) or (len(tdata[bdata == "g"]) == 0):
        return None
    
    max_flux = np.max(fdata[bdata == "r"] - np.abs(ferrdata[bdata == "r"]))
    max_flux_loc =  tdata[np.argmax(fdata[bdata == "r"] - np.abs(ferrdata[bdata == "r"]))]
    inc_band_ix = (bdata == "g")

    
    def jax_model(t = None, obsflux = None, uncertainties = None, max_flux = None, t0_approx = None, inc_band_ix = None):
        
        with numpyro.plate('components', 1):
            A = max_flux * 10**numpyro.sample("logA", trunc_norm(*PRIOR_A))
            beta = numpyro.sample("beta", trunc_norm(*PRIOR_BETA))
            gamma = 10**numpyro.sample("log_gamma", trunc_norm(*PRIOR_GAMMA))
            t0 = numpyro.sample("t0", trunc_norm(np.amin(t) - 50., np.amax(t) + 50., t0_approx, 20.))
            tau_rise = 10**numpyro.sample("log_tau_rise", trunc_norm(*PRIOR_TAU_RISE))
            tau_fall = 10**numpyro.sample("log_tau_fall", trunc_norm(*PRIOR_TAU_FALL))
            extra_sigma = 10**numpyro.sample("log_extra_sigma", trunc_norm(*PRIOR_EXTRA_SIGMA))
            
            A_g = numpyro.sample("A_g", trunc_norm(*PRIOR_A_g))
            beta_g = numpyro.sample("beta_g", trunc_norm(*PRIOR_BETA_g))
            gamma_g = numpyro.sample("gamma_g", trunc_norm(*PRIOR_GAMMA_g))
            t0_g = numpyro.sample("t0_g", trunc_norm(*PRIOR_T0_g))
            tau_rise_g = numpyro.sample("tau_rise_g", trunc_norm(*PRIOR_TAU_RISE_g))
            tau_fall_g = numpyro.sample("tau_fall_g", trunc_norm(*PRIOR_TAU_FALL_g))
            extra_sigma_g = numpyro.sample("extra_sigma_g", trunc_norm(*PRIOR_EXTRA_SIGMA_g))

        A_b = A * A_g
        beta_b = beta * beta_g
        gamma_b = gamma * gamma_g
        t0_b = t0 * t0_g
        tau_rise_b = tau_rise * tau_rise_g
        tau_fall_b = tau_fall * tau_fall_g
        
        phase = t - t0    
        flux_const = A / (1. + jnp.exp(-phase / tau_rise))
        sigmoid = 1 / (1 + jnp.exp(gamma-phase / 10.))
        
        flux = flux_const * ( (1-sigmoid) * (1 - beta*phase) + sigmoid * (1 - beta*gamma) * jnp.exp(-(phase-gamma)/tau_fall) )

        # g band
        phase_b = (t - t0_b)[inc_band_ix]
        flux_const_b = A / (1. + jnp.exp(-phase_b / tau_rise_b))
        sigmoid_b = 1 / (1 + jnp.exp(gamma_b-phase_b / 10.))
        
        flux = flux.at[inc_band_ix].set(flux_const_b * ( (1-sigmoid_b) * (1 - beta_b*phase_b) + sigmoid_b * (1 - beta_b*gamma_b) * jnp.exp(-(phase_b-gamma_b)/tau_fall_b) ))
        
        sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)
        sigma_tot = sigma_tot.at[inc_band_ix].set(jnp.sqrt(uncertainties[inc_band_ix]**2 + extra_sigma_g**2 * extra_sigma**2))
        
        with numpyro.plate("obsflux", len(obsflux)):
            obs = numpyro.sample("obs",dist.Normal(flux, sigma_tot),obs=obsflux)
            
    kernel = NUTS(jax_model)
    num_samples = 10000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, num_chains=1)
    with numpyro.validation_enabled():
        res = mcmc.run(rng_key, obsflux = fdata, t=tdata, uncertainties=ferrdata, max_flux=max_flux, t0_approx=max_flux_loc, inc_band_ix=inc_band_ix)
        
    mcmc.print_summary()
    posterior_samples = mcmc.get_samples()
    predictive = Predictive(jax_model, posterior_samples, infer_discrete=True)
    discrete_samples = predictive(random.PRNGKey(1), sn_ids = sn_ids, p_sns=p_sns,obsflux = datas, t=times, uncertainties = uncertainties, )
    plt.hist(posterior_samples['mean_tfall'].T,bins=100)
    plt.savefig('test.png')
    sys.exit()
    az.plot_trace(posterior_samples, compact=True);
    plt.savefig('test.png')

    """
    if plot:
        plt.errorbar(tdata[bdata=="g"], fdata[bdata=="g"], yerr=ferrdata[bdata=="g"], c="g", label="g", fmt="o")
        plt.errorbar(tdata[bdata=="r"], fdata[bdata=="r"], yerr=ferrdata[bdata=="r"], c="r", label="r", fmt="o")

        trange_fine = np.linspace(np.amin(tdata), np.amax(tdata), num=500)

        for sample in eq_wt_samples[:30]:
            plt.plot(trange_fine, flux_model(sample, trange_fine, ["g"] * len(trange_fine)), c="g", lw=1, alpha=0.1)
            plt.plot(trange_fine, flux_model(sample, trange_fine, ["r"] * len(trange_fine)), c="r", lw=1, alpha=0.1)

        plt.xlabel("MJD")
        plt.ylabel("Flux")
        plt.title(prefix)
        if t0_lim is None:
            plt.savefig(os.path.join(FIT_PLOTS_FOLDR, prefix+".png"))
        else:
            plt.savefig(os.path.join(FIT_PLOTS_FOLDR, prefix+"_%.02f.png" % t0))
        plt.close()
    """

    return posterior_samples


def main_loop_single_file(test_fn, output_dir=FITS_DIR):
    #try:
    os.makedirs(output_dir, exist_ok=True)
    prefix = test_fn.split("/")[-1][:-4]
    #if os.path.exists(output_dir + str(prefix) + '_eqwt.npz'):
    #    return None

    base_band_i = 1 # second of g, r band base fit
    eq_samples = run_mcmc(test_fn)
    if eq_samples is None:
        return None
    print(np.mean(eq_samples, axis=0))
    prefix = test_fn.split("/")[-1][:-4]

    np.savez_compressed(output_dir + str(prefix) + '_eqwt.npz', eq_samples)
    #except:
    #    print("skipped")
    #    return None
    
    
if __name__ == "__main__":
    main_loop_single_file(os.path.join(DATA_DIRS[0],"ZTF22aaaehzu.npz"))
