"""Run fits for original Superphot pipeline."""
import sys, os
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false --xla_force_host_platform=true"
os.environ['PYTENSOR_FLAGS']=f'compiledir_format=compiler2'

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)   # Use 64-bit precision for better numerical stability
jax.config.update('jax_platform_name', 'cpu')  # Use CPU platform

import pytensor
#pytensor.config.compiledir_format = "compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s-%(device)s"

from astropy.table import Table
from astropy.io import ascii
from superphot.fit import two_iteration_mcmc
from superphot.extract import *
import pandas as pd
from contextlib import contextmanager

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import convert_mags_to_flux
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples
from multiprocess import Pool
import glob

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
            
def fit_lcs_superphot(dataset_csv, probs_csv, data_dir, save_dir):
    """Run superphot-fit on processed light curves."""
    print("STARTS")
    os.makedirs(save_dir, exist_ok=True)
    
    full_df = pd.read_csv(dataset_csv)
    all_names = full_df.NAME.to_numpy()
    labels = full_df.CLASS.to_numpy()
    redshifts = full_df.Z.to_numpy()
        
    final_names = pd.read_csv(probs_csv).Name.to_numpy()

    def single_fit(i):
        name = all_names[i]
        
        if name != 'ZTF23aauqmys':
            return
        if name not in final_names:
            print(name)
            return
        
        l_canon = SnClass.canonicalize(labels[i])
            
        lc = Lightcurve.from_file(
            os.path.join(
                data_dir,
                name + ".npz"
            )
        )
        
        df = pd.DataFrame(
            {
                "PHASE": lc.times,
                "FLUXCAL": lc.fluxes,
                "FLUXCALERR": lc.flux_errors,
                "FLT": np.where(lc.bands == 'r', 'r', 'g')
            }
        )
        
        lc = Table.from_pandas(df)
        lc.meta = {
            'SNID': name,
            'REDSHIFT':  redshifts[i],
            'FILTERS': 'gr',
        }
        
        outfile = os.path.join(
            save_dir,
            name + '{}'
        )
        #with suppress_stdout():
        two_iteration_mcmc(
            lc,
            outfile,
            do_diagnostics=False,
            force=True,
        )
    p = Pool(8)
    result = p.map(single_fit, np.arange(len(all_names)))
    """
    with Pool() as pool:
        result = pool.map(
            single_fit,
            np.arange(len(all_names))
        )
    """

def create_metatable(
    full_csv,
    probs_csv,
    save_path
):
    """Create metatable to use in Superphot.
    """
    df = pd.read_csv(full_csv)
    final_names = pd.read_csv(probs_csv).Name.to_numpy()
    df_red = df[df.NAME.isin(final_names)]
    metatable_df = pd.DataFrame({
        "filename": df_red.NAME.to_numpy(),
        "redshift": df_red.Z.to_numpy(),
        "MWEBV": np.zeros(len(final_names)),
        "type": df_red.CLASS.to_numpy()
    })
    metatable = Table.from_pandas(metatable_df)
    ascii.write(metatable, save_path, overwrite=True)  
    
def encode_superphot_features(
    lc_file,
    metatable_fn,
    paramtable_fn,
    save_dir
):
    """Encode Superphot files as PosteriorSamples
    objects.
    """
    data_table = compile_parameters(
        lc_file,
        ['r', 'g'],
        ndraws=100,
    )
    np.savez_compressed(
        paramtable_fn,
        **data_table,
        **data_table.meta
    )
    t = load_data(metatable_fn, paramtable_fn)
    data = extract_features(
        t, zero_point=26.3,
        use_median=False, use_pca=True,
        stored_pcas=None, save_pca_to=None,
        save_reconstruction_to=None
    )
    
    names = data['filename']
    features = data['features']
    labels = data['type']
    redshifts = data['redshift']
    
    for i, name in enumerate(names):
        features_i = features[i]
        
        ps = PosteriorSamples(
            features_i,
            name=name,
            sampling_method='superphot',
            redshift=redshifts[i],
            sn_class=labels[i]
        )
        ps.save_to_file(save_dir)

        

    