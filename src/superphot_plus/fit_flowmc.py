import numpy as np
import jax
import jax.numpy as jnp
from jax import jit
from jax.config import config

from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.realNVP import RealNVP
#from flowMC.sampler.MALA import MALA
from flowMC.sampler.HMC import HMC
from flowMC.sampler.Sampler import Sampler
from flowMC.nfmodel.rqSpline import MaskedCouplingRQSpline

import os
import matplotlib.pyplot as plt
import corner

from constants import *
from custom_mala import MALA

config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)

NCHAINS = 4
SEED = 42


def import_data(filename, t0_lim=None):
    """Import the data file.
    
    Parameters
    ----------
    filename : str
        The name of the data file to import.
    t0_lim : float or None, optional
        The upper time limit for the data. If provided, only data points
        with time values less than or equal to t0_lim will be included.
        Defaults to None.

    Returns
    -------
    tuple of np.ndarray or None
        A tuple containing the padded time, flux, flux error, and band 
        arrays, respectively. If the input data does not contain any 
        valid points, None is returned.

    """
    npy_array = np.load(filename)
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

    if (t[b == "r"] is None) or (len(t[b == "r"]) == 0):
        return None
    if (t[b == "g"] is None) or (len(t[b == "g"]) == 0):
        return None

    b = np.where(b == "r", 1, 0) # change to integers

    # sort
    sort_idx = np.argsort(t)
    t = t[sort_idx]
    f = f[sort_idx]
    ferr = ferr[sort_idx]
    b = b[sort_idx]

    max_flux_loc =  t[b == 1][np.argmax(f[b == 1] - np.abs(ferr[b == 1]))]

    t -= max_flux_loc # make relative

    # separate r and g band points
    # necessary for static indexing for jax

    # pad data
    t_padded, f_padded, ferr_padded, b_padded = (
        np.array([]),
        np.array([]),
        np.array([]),
        np.array([]),
    )

    for b_int in [0,1]:
        len_b = len(b[b == b_int])
        t_s = t[b == b_int]
        f_s = f[b == b_int]
        ferr_s = ferr[b == b_int]
        b_s = b[b == b_int]

        if len_b > PAD_SIZE:
            t_padded = np.append(t_padded, t_s[:PAD_SIZE])
            f_padded = np.append(f_padded, f_s[:PAD_SIZE])
            ferr_padded = np.append(ferr_padded, ferr_s[:PAD_SIZE])
            b_padded = np.append(b_padded, b_s[:PAD_SIZE])
        else:
            t_padded = np.append(t_padded, t_s)
            f_padded = np.append(f_padded, f_s)
            ferr_padded = np.append(ferr_padded, ferr_s)
            b_padded = np.append(b_padded, b_s)

            t_padded = np.append(t_padded, [5000] * (PAD_SIZE - len_b))
            f_padded = np.append(f_padded, [0.] * (PAD_SIZE - len_b))
            ferr_padded = np.append(ferr_padded, [1e10] * (PAD_SIZE - len_b))
            b_padded = np.append(b_padded, [b_int] * (PAD_SIZE - len_b) )

    return t_padded, f_padded, ferr_padded, b_padded

@jit
def prior_eval(cube, _):
    """
    From a parameter cube, evaluates the associated
    prior probability.
    """
    return -0.5 * jnp.sum((cube - PRIOR_MEANS)**2 / PRIOR_SIGMAS**2)
    
@jit
def posterior_eval(cube, data_stacked):
    """
    Extracts the parameter cube and evaluates
    the associated likelihood.
    """
    #return -0.5 * jnp.linalg.norm((cube - PRIOR_MEANS - 1.)/PRIOR_SIGMAS)
    
    t, obsflux, uncertainties = data_stacked

    #return -50. * jnp.linalg.norm(cube - 100.)
    
    max_flux = np.max( obsflux - uncertainties )

   
    #return prior_eval(cube, max_flux)

    A, beta, gamma, t0, tau_rise, tau_fall, extra_sigma, \
        A_g, beta_g, gamma_g, t0_g, tau_rise_g, tau_fall_g, \
        extra_sigma_g = cube

    A = max_flux * 10**A
    gamma = 10**gamma
    tau_rise = 10**tau_rise
    tau_fall = 10**tau_fall
    extra_sigma = 10**extra_sigma
    
    phase = t - t0
    flux_const = A / (1. + jnp.exp(-phase / tau_rise))
    
    sigmoid = 1 / (1 + jnp.exp(10.*(gamma - phase)))

    #return -jnp.sum( (obsflux[:14] - flux_const)**2 )
    flux = flux_const * (
        (1 - sigmoid) * (1 - beta * phase)
        + sigmoid * (1 - beta * gamma) * jnp.exp(-(phase - gamma) / tau_fall)
    )
    

    inc_band_ix = np.arange( 0, PAD_SIZE )

    A_b = A * A_g # pylint: disable=unused-variable
    beta_b = beta * beta_g
    gamma_b = gamma * gamma_g
    t0_b = t0 * t0_g
    tau_rise_b = tau_rise * tau_rise_g
    tau_fall_b = tau_fall * tau_fall_g

    

    # g band
    phase_b = (t - t0_b)[inc_band_ix]
    flux_const_b = A / (1. + jnp.exp(-phase_b / tau_rise_b))
    sigmoid_b = 1 / (1 + jnp.exp(10.*(gamma_b - phase_b)))
    
    flux = flux.at[inc_band_ix].set(
        flux_const_b
        * (
            (1 - sigmoid_b) * (1 - beta_b * phase_b)
            + sigmoid_b
            * (1 - beta_b * gamma_b)
            * jnp.exp(-(phase_b - gamma_b) / tau_fall_b)
        )
    )


    sigma_tot = jnp.sqrt(uncertainties**2 + extra_sigma**2)
    sigma_tot = sigma_tot.at[inc_band_ix].set(
        jnp.sqrt(uncertainties[inc_band_ix] ** 2 + extra_sigma_g**2 * extra_sigma**2)
    )

    return prior_eval(cube, None) - 0.5 * jnp.sum((flux - obsflux)**2 / sigma_tot**2) - jnp.sum( jnp.log( jnp.sqrt(2 * jnp.pi) * sigma_tot ) )

    
def run_flowMC(lc_filename):
    """
    Run flowMC on one light curve.
    """
    tdata, fdata, ferrdata, bdata = import_data(lc_filename, t0_lim=None)
    
    if tdata is None:
        return None

    max_flux = np.max( fdata - ferrdata )
    print(max_flux)
    data_stacked = jnp.array([tdata, fdata, ferrdata])
    
    n_chains = NCHAINS
    rng_key_set = initialize_rng_keys(n_chains, seed=SEED)
    n_dim = 14
    initial_position = jnp.tile(PRIOR_MEANS, (n_chains, 1))
    #initial_position[:,0] = max_flux * 10.0**initial_position[:,0]
    #initial_position[:,[2,4,5,6]] = 10.0**initial_position[:,[2,4,5,6]]
    
    print(jax.value_and_grad(prior_eval)(initial_position[0], None))
    print(jax.value_and_grad(posterior_eval)(initial_position[0], data_stacked))
    #return
    
    n_layer = 10  # number of coupling layers
    n_hidden = 128  # with of hidden layers in MLPs parametrizing coupling layers

    #model = RealNVP(n_layer, n_dim, n_hidden, jax.random.PRNGKey(10))
    model = MaskedCouplingRQSpline(n_dim, 4, [32,32], 8 , jax.random.PRNGKey(10))
    step_size = 1e-1
    
    n_loop_training = 10
    n_loop_production = 10
    n_local_steps = 100
    n_global_steps = 100
    num_epochs = 10

    learning_rate = 0.005
    momentum = 0.9
    batch_size = 500
    max_samples = 500


    sampler = MALA(posterior_eval, True, {"step_size": PRIOR_SIGMAS / 100.}, use_autotune=True)#{"})
    #sampler = HMC(posterior_eval, True, {"step_size": 0.1, "n_leapfrog": 4, "inverse_metric": 1}, verbose=True)
    nf_sampler = Sampler(
        n_dim=n_dim,
        rng_key_set=rng_key_set,
        local_sampler=sampler,
        data=data_stacked,
        nf_model=model,
        n_loop_training=n_loop_training,
        n_loop_production=n_loop_production,
        n_local_steps=n_local_steps,
        n_global_steps=n_global_steps,
        n_chains=n_chains,
        n_epochs=num_epochs,
        learning_rate=learning_rate,
        momentum=momentum,
        batch_size=batch_size,
        use_global=True,
    )
    
    nf_sampler.sample(initial_position, data_stacked)
    
    out_prod = nf_sampler.get_sampler_state()   # default training=False
    out_train = nf_sampler.get_sampler_state(training=True)
    chains = np.reshape(out_prod['chains'][:,-100:], (100*n_chains, n_dim))
    
    all_chains = np.array(out_train['chains'])
    print(all_chains)
    global_accs = np.array(out_train['global_accs'])
    local_accs = np.array(out_train['local_accs'])
    loss_vals = np.array(out_train['loss_vals'])
    nf_samples = np.array(nf_sampler.sample_flow(1000)[1])


    # Plot 2 chains in the plane of 2 coordinates for first visual check 
    plt.figure(figsize=(6, 6))
    axs = [plt.subplot(2, 2, i + 1) for i in range(4)]
    plt.sca(axs[0])
    plt.title("2d proj of 2 chains")

    plt.plot(all_chains[0, :, 0], all_chains[0, :, 1], 'o-', alpha=0.5, ms=2)
    plt.plot(all_chains[1, :, 0], all_chains[1, :, 1], 'o-', alpha=0.5, ms=2)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

    plt.sca(axs[1])
    plt.title("NF loss")
    plt.plot(loss_vals.reshape(-1))
    plt.xlabel("iteration")

    plt.sca(axs[2])
    plt.title("Local Acceptance")
    plt.plot(local_accs.mean(0))
    plt.xlabel("iteration")

    plt.sca(axs[3])
    plt.title("Global Acceptance")
    plt.plot(global_accs.mean(0))
    plt.xlabel("iteration")
    plt.tight_layout()
    plt.savefig("../../../test_1.png")
    plt.close()

    labels=["$x_1$", "$x_2$", "$x_3$", "$x_4$", "$x_5$"]
    # Plot all chains
    figure = corner.corner(
        all_chains.reshape(-1, n_dim), labels=labels
    )
    figure.set_size_inches(7, 7)
    figure.suptitle("Visualize samples")
    plt.savefig("../../../test_2.png")
    plt.close()

    # Plot Nf samples
    figure = corner.corner(nf_samples, labels=labels)
    figure.set_size_inches(7, 7)
    figure.suptitle("Visualize NF samples")
    plt.savefig("../../../test_3.png")
    plt.close()

    
    return chains


def flowMC_single_file(filename, output_dir):
    """
    Run flowMC on a single light curve.
    """
    os.makedirs(output_dir, exist_ok=True)

    eq_samples = run_flowMC(filename)

    if eq_samples is None:
        return None

    print(np.mean(eq_samples, axis=0))
    prefix = filename.split("/")[-1][:-4]

    np.savez_compressed(os.path.join(output_dir , f"{prefix}_eqwt_flowMC.npz" ), eq_samples)
    
    return None


if __name__ == "__main__":
    test_fn = "../../tests/data/ZTF22abvdwik.npz"
    flowMC_single_file(test_fn, "../../..")

    
    

    
    
    