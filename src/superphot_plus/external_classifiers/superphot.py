"""Run fits for original Superphot pipeline."""
from superphot.fit import two_iteration_mcmc
from superphot.extract import *
import pandas as pd
import os
import numpy as np
from astropy.table import Table
from astropy.io import ascii

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import convert_mags_to_flux
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples


def fit_lcs_superphot(dataset_csv, probs_csv, data_dir, save_dir):
    """Run superphot-fit on processed light curves."""
    print("STARTS")
    os.makedirs(save_dir, exist_ok=True)
    
    full_df = pd.read_csv(dataset_csv)
    all_names = full_df.NAME.to_numpy()
    labels = full_df.CLASS.to_numpy()
    redshifts = full_df.Z.to_numpy()
        
    final_names = pd.read_csv(probs_csv).Name.to_numpy()
    lcs = []
    
    for i, name in enumerate(all_names):
        if i % 500 == 0:
            print(i)
            
        if name not in final_names:
            continue
        
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
        
        two_iteration_mcmc(
            lc,
            outfile,
            do_diagnostics=False,
            force=False,
        )

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

        

    