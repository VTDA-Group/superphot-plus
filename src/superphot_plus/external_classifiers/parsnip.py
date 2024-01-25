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

DEVICE = "mps"

DEFAULT_FEATURES = [
    'color',
    's1',
    's2',
    's3',
    'luminosity',
    'reference_time'
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
        
    print(len(lcs))
    dataset = lcdata.from_light_curves(lcs)
    dataset.write_hdf5(
        os.path.join(
            save_dir,
            'parsnip_dataset.h5'
        ),
        overwrite=True
    )
    
def train_parsnip_model(
    dataset_hdf5,
    model_prefix,
):
    """Train ParSNIP model on custom device."""
    dataset = parsnip.load_datasets(
        [dataset_hdf5,],
        require_redshift=True
    )
    bands = parsnip.get_bands(dataset)
    #settings = parsnip.default_settings
    #settings['bands'] = bands
    model = parsnip.ParsnipModel(
        model_prefix + '.pt',
        bands,
        device=DEVICE,
        settings={},
        ignore_unknown_settings=True
    )

    dataset = model.preprocess(dataset)

    train_dataset, test_dataset = parsnip.split_train_test(dataset)
    model.fit(train_dataset, test_dataset=test_dataset, max_epochs=1000)
    # Save the score to a file for quick comparisons. If we have a small dataset,
    # repeat the dataset several times when calculating the score.
    rounds = int(np.ceil(25000 / len(train_dataset)))
    train_score = model.score(train_dataset, rounds=rounds)
    test_score = model.score(test_dataset, rounds=10 * rounds)
    end_time = time.time()

    # Time taken in minutes
    elapsed_time = (end_time - start_time) / 60.

    with open(model_prefix + '.log', 'a') as f:
        print(
            f'{model_path} {model.epoch} {elapsed_time:.2f} {train_score:.4f} '
              f'{test_score:.4f}', file=f
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


def encode_with_parsnip(
    light_curves,
    labels,
    redshifts,
    model_path,
    save_dir,
    csv_path,
    feature_list=DEFAULT_FEATURES,
):
    """Retrieve ParSNIP features and format
    to be read by get_posterior_samples.
    
    Parameters
    ----------
    light_curves: list of Lightcurve
        The light curves to encode into a
        PosteriorSamples object
    model_path: str
        filepath of parsnip model
    save_dir: str
        where to save posterior files.
    csv_path: str
        where to save training CSV
    feature_list: list of str
        which features to save
    """
    model = parsnip.load_model(model_path, device=DEVICE)
    names = []
    ls = []
    zs = []
    
    lcs = []
        
    for i, lc in enumerate(light_curves):
        df = pd.DataFrame(
            {
                "mjd": lc.times,
                "flux": lc.fluxes,
                "fluxerr": lc.flux_errors,
                "bandpass": np.where(lc.bands == 'r', 'ztfr', 'ztfg')
            }
        )
        
        lc_parsnip = Table.from_pandas(df)
        l_canon = SnClass.canonicalize(labels[i])
        lc_parsnip.meta = {
            'id': lc.name,
            #'right_ascension': np.mean(sub_df.ra.to_numpy()),
            #'decl': np.mean(sub_df.dec.to_numpy()),
            'class': l_canon,
            'redshift': redshifts[i]
        }
        lcs.append(lc_parsnip)
    
    dataset = lcdata.from_light_curves(lcs)
    predictions = model.predict_dataset(dataset, augment=False)
    features_mu = np.asarray([predictions[k] for k in feature_list]).T
    
    print(features_mu.shape)
    if 'reference_time' in feature_list:
        ref_time_idx = feature_list.index('reference_time')
        features_mu[:,ref_time_idx] = 0
        
    print(features_mu.shape)
    features_sigma = np.asarray([predictions[k+"_error"] for k in feature_list]).T
    
    labels_sorted = np.asarray(predictions['type'])
    redshifts_sorted = np.asarray(predictions['redshift'])
    
    for i, name in enumerate(predictions['object_id']):
        
        features_i = np.array(
            [
                np.random.normal(m, s, 100) for m, s in zip(
               features_mu[i], features_sigma[i]
                )
            ]
        ).T  # draw 100 samples from distribution
        
        ps = PosteriorSamples(
            features_i,
            name=name,
            sampling_method='parsnip',
            redshift=redshifts_sorted[i],
            sn_class=labels_sorted[i]
        )
        ps.save_to_file(save_dir)
        
        
    # make training CSV
    training_df = pd.DataFrame({
        'NAME': np.asarray(predictions['object_id']),
        'CLASS': labels_sorted,
        'Z': redshifts_sorted
    })
    training_df.to_csv(csv_path)
    
    
        
        
    