import parsnip
import lcdata
import pandas as pd
import os
import numpy as np
from astropy.table import Table

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import convert_mags_to_flux
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples


DEFAULT_FEATURES = [
    'color',
    'color_error',
    's1',
    's1_error',
    's2',
    's2_error',
    's3',
    's3_error',
    'luminosity',
    'luminosity_error',
    'reference_time_error',
]
    
def convert_to_lcdata_h5(dataset_csv, probs_csv, data_dir, save_dir):
    """Convert data folder to h5 file compatible with lcdata.
    """
    labels_to_classes, _ = SnClass.get_type_maps()
    
    full_df = pd.read_csv(dataset_csv)
    all_names = full_df.NAME.to_numpy()
    labels = full_df.CLASS.to_numpy()
    redshifts = full_df.Z.to_numpy()
    
    probs_names = pd.read_csv(probs_csv).Name.to_numpy()
    
    lcs = []
    
    for i, name in enumerate(all_names):
        if i % 500 == 0:
            print(i)
            
        if name not in probs_names:
            continue
            
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
                "mjd": lc.times,
                "flux": lc.fluxes,
                "fluxerr": lc.flux_errors,
                "bandpass": np.where(lc.bands == 'r', 'ztfr', 'ztfg')
            }
        )
        
        lc = Table.from_pandas(df)
        lc.meta = {
            'id': name,
            #'right_ascension': np.mean(sub_df.ra.to_numpy()),
            #'decl': np.mean(sub_df.dec.to_numpy()),
            'class': labels_to_classes[l_canon],
            'redshift': redshifts[i]
        }
        
        lcs.append(lc)
        
    dataset = lcdata.from_light_curves(lcs)
    dataset.write_hdf5(
        os.path.join(
            save_dir,
            'parsnip_dataset.h5'
        ),
        overwrite=True
    )
    
def retrieve_decodings(sn_name, dataset_fn, model_fn):
    """Retrieve 100 decoded light curves for single
    light curve name."""
    
    dataset = lcdata.read_hdf5(dataset_fn)
    model = parsnip.load_model(model_fn)
    light_curve = dataset.get_lc(sn_name)
    model_times, model_flux, model_result = model.predict_light_curve(
        light_curve, sample=True, count=100
    )
    percentile_offset = (100 - 68.) / 2.
    flux_median = np.median(model_flux, axis=0)
    flux_min = np.percentile(model_flux, percentile_offset,
                             axis=0)
    flux_max = np.percentile(model_flux,
                             100 - percentile_offset, axis=0)
    
    return model_times, flux_median, flux_min, flux_max


def reformat_features(
    predictions_fn,
    save_dir,
    csv_path,
    feature_list=DEFAULT_FEATURES,
):
    """Retrieve ParSNIP features and format
    to be read by get_posterior_samples.
    """
    predictions = Table.read(predictions_fn)
    features = np.array([
        predictions[i].data for i in feature_list
    ]).T
    names = predictions['object_id'].data.astype(str)
    redshifts = predictions['redshift'].data.astype(float)
    labels = predictions['type'].data.astype(str)
    
    # make training CSV
    training_df = pd.DataFrame({
        'NAME': names,
        'CLASS': labels,
        'Z': redshifts
    })
    training_df.to_csv(csv_path)
    for i, sn_name in enumerate(names):
        ps = PosteriorSamples(
            features[i],
            name=sn_name,
            sampling_method='parsnip',
            redshift=redshifts[i],
            sn_class=labels[i]
        )
        ps.save_to_file(save_dir)
    
    
        
        
    