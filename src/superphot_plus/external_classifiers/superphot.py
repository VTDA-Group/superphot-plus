"""Run fits for original Superphot pipeline."""
from superphot.fit import two_iteration_mcmc
import pandas as pd
import os
import numpy as np
from astropy.table import Table

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import convert_mags_to_flux
from superphot_plus.lightcurve import Lightcurve


def fit_lcs_superphot(dataset_csv, data_dir, save_dir):
    """Run superphot-fit on processed light curves."""
    print("STARTS")
    os.makedirs(save_dir, exist_ok=True)
    
    full_df = pd.read_csv(dataset_csv)
    all_names = full_df.NAME.to_numpy()
    labels = full_df.CLASS.to_numpy()
    redshifts = full_df.Z.to_numpy()
        
    lcs = []
    
    for i, name in enumerate(all_names):
        if i % 500 == 0:
            print(i)
            
        if np.isnan(redshifts[i]):
            print(name)
        
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
            'SNID': 'PS0909006',
            'REDSHIFT':  redshifts[i],
            'FILTERS': 'gr',
        }
        
        outfile = os.path.join(
            save_dir,
            name + '{}'
        )
        
        traces1, traces2, parameters = two_iteration_mcmc(
            lc,
            outfile,
            do_diagnostics=False,
            force=False,
        )
        

    