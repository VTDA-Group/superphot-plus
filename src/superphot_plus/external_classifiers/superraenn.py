"""Run SuperRAENN on superphot+ light curves."""
import pandas as pd
import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck13 as cosmo

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import convert_mags_to_flux
from superphot_plus.lightcurve import Lightcurve
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.file_utils import get_posterior_filename
from tensorflow.keras.optimizers.legacy import Adam

from superraenn.preprocess import save_lcs
from superraenn.lc import LightCurve
from superraenn.raenn import *
from superraenn.feature_extraction import *
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, RepeatVector, concatenate
import tensorflow as tf

NEURON_N_DEFAULT = 100
ENCODING_N_DEFAULT = 10
N_EPOCH_DEFAULT = 2000
    

def prep_lcs_superraenn(
    dataset_csv,
    probs_csv,
    data_dir,
    save_dir,
    metafile
):
    """Run equivalent of superraenn-prep on processed light curves."""
    print("STARTS")
    os.makedirs(save_dir, exist_ok=True)
    
    full_df = pd.read_csv(dataset_csv)
    all_names = full_df.NAME.to_numpy()
    labels = full_df.CLASS.to_numpy()
    redshifts = full_df.Z.to_numpy()
    
    final_names = pd.read_csv(probs_csv).Name.to_numpy()
        
    my_lcs = []
    
    with open(metafile, 'w+') as f:
        f.write('# SN Redshift Type T_explosion MW(EBV)\n')

    for i, name in enumerate(all_names):
        if i % 500 == 0:
            print(i)
            
        if name not in final_names:
            continue
            
        l_canon = SnClass.canonicalize(labels[i])
        l_oneword = l_canon.replace(" ", "")
            
        lc = Lightcurve.from_file(
            os.path.join(
                data_dir,
                name + ".npz"
            )
        )

        my_lc = LightCurve(
            name,
            lc.times[lc.bands != 'i'],
            lc.fluxes[lc.bands != 'i'],
            lc.flux_errors[lc.bands != 'i'],
            lc.bands[lc.bands != 'i']
        )
        
        filt_dict = {'g': 0, 'r': 1}

        my_lc.add_LC_info(
            zpt=26.3,
            redshift=redshifts[i],
            lim_mag=20.6,
            obj_type=l_canon
        )
        my_lc.get_abs_mags()
        my_lc.sort_lc()
        my_lc.correct_time_dilation()
        my_lc.filter_names_to_numbers(filt_dict)
        my_lc.cut_lc()
        
        try:
            my_lc.make_dense_LC(2)
            my_lcs.append(my_lc)
            
            with open(metafile, 'a') as f:
                f.write(f'{name} {redshifts[i]} {l_oneword} 0.0 0.0\n')
                
        except:
            print("skipped")
            continue
        
    save_lcs(my_lcs, save_dir)
    
    
def run_superraenn_raenn(lcfile, outdir, load_file=None):

    sequence, outseq, ids, maxlen, nfilts = prep_input(lcfile, save=True, outdir=outdir)

    model, callbacks_list, input_1, encoded = make_model(
        NEURON_N_DEFAULT,
        ENCODING_N_DEFAULT,
        int(maxlen),
        2
    )
    
    if load_file is not None:
        model = tf.keras.models.load_model(
            load_file,
            custom_objects={'customLoss': customLoss},
            compile=False
        )
        
    new_optimizer = Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, decay=0)
    model.compile(optimizer=new_optimizer, loss=customLoss)
    model = fit_model(model, callbacks_list, sequence, outseq, N_EPOCH_DEFAULT)
    encoder = get_encoder(model, input_1, encoded)

    # These comments used in testing, and sould be removed...
    # lms = outseq[:, 0, 1]
    # test_model(sequence_test, model, lm, maxlen, plot=True)
    # decoder = get_decoder(model, args.encodingN)
    # get_decodings(decoder, encoder, sequence, lms, args.encodingN, \
    #               maxlen, plot=False)

    if outdir[-1] != '/':
        outdir += '/'
        
    save_model(model, ENCODING_N_DEFAULT, NEURON_N_DEFAULT, outdir=outdir)

    save_encodings(model, encoder, sequence, ids, lcfile,
                   ENCODING_N_DEFAULT, NEURON_N_DEFAULT, len(ids), maxlen,
                   outdir=outdir)


def encode_raenn_features(
    lc_file,
    model_base,
    prep_file,
    save_dir,
    raenn_only=False
):
    """Encode RAENN features in PosteriorSamples objects.
    """
    input_lcs = np.load(lc_file, allow_pickle=True)['lcs']
    ids = []
    feat_names = []
    for input_lc in input_lcs:
        ids.append(input_lc.name)
    feat = feat_from_raenn(
        lc_file,
        model_base=model_base,
        prep_file=prep_file,
        plot=False
    )
    features = feat
    if raenn_only:
        for i, input_lc in enumerate(input_lcs):
            ps = PosteriorSamples(
                features[i],
                name=input_lc.name,
                sampling_method='superraenn',
                redshift=input_lc.redshift,
                sn_class=input_lc.obj_type
            )
            ps.save_to_file(save_dir)
            
    
    else:
        nfilts = 2
        feats_all = []
        for i, input_lc in enumerate(input_lcs):
            if i % 100 == 0:
                print(i)
            save_path = get_posterior_filename(
                input_lc.name, fits_dir=save_dir, sampler='superraenn'
            )
            #if os.path.exists(save_path):
            #    continue
            gp = input_lc.gp
            gp_mags = input_lc.gp_mags
            
            feats = []
            
            for j in np.arange(nfilts):
                new_times = np.linspace(-100, 100, 500)
                x_stacked = np.asarray([new_times, [j] * 500]).T
                pred, var = gp.predict(gp_mags, x_stacked)
                
                feats.append(
                    np.nanmin(input_lc.dense_lc[:, j, 0], axis=0)
                )
                max_ind = np.nanargmin(pred)
                max_mag = pred[max_ind]
                max_t = new_times[max_ind]
                
                for n_mag in [1,2,3]:
                    trise = np.where(
                        (new_times < max_t) & (pred > (max_mag + n_mag))
                    )
                    tfall = np.where(
                        (new_times > max_t) & (pred > (max_mag + n_mag))
                    )
                    if len(trise[0]) == 0:
                        trise = max_t - np.min(new_times)
                    else:
                        trise = max_t - new_times[trise][-1]

                    if len(tfall[0]) == 0:
                        tfall = np.max(new_times) - max_t
                    else:
                        tfall = new_times[tfall][0] - max_t
                    
                    feats.append(trise)
                    feats.append(tfall)
                
                
                new_times_sub = new_times - max_t
                lc_grad = np.gradient(pred, new_times_sub)
                gindmean = np.where(
                    (
                        new_times_sub > 10
                    ) & (
                        new_times_sub < 30
                    )
                )
                feats.append(
                    np.nanmedian(lc_grad[gindmean])
                )
                feats.append(np.trapz(pred))
            features_i = np.append(features[i], feats)
            ps = PosteriorSamples(
                features_i,
                name=input_lc.name,
                sampling_method='superraenn',
                redshift=input_lc.redshift,
                sn_class=input_lc.obj_type
            )
            ps.save_to_file(save_dir)

def retrieve_decodings(
    model_path,
    lc_file,
    prep_fn,
    sn_name=None
):
    
    model = load_model(
        model_path,
        custom_objects={"customLoss": customLoss},
        compile=False
    )
    input_1 = Input((None, 5))
    mid_layer = model.layers[1](input_1)
    encoded = model.layers[2](mid_layer)
    
    sequence, outsequence, ids, maxlen, nfilts = prep_input(
        lc_file, save=False,
    )
    
    ids = np.array(ids)
    
    if sn_name is not None:
        seq = sequence[ids == sn_name]
        outseq = outsequence[ids == sn_name]
        outseq_single = outsequence[ids == sn_name]
        u_time = np.unique(outseq_single[0,:,:])
        outseq = np.zeros((1, 200, 2))
        outseq[0, :, 0] = np.linspace(
            np.min(u_time[:-1]),
            np.max(u_time[:-1]),
            num=200
        )
        outseq = outseq_single
        outseq[0, :, 1] = outseq_single[0,0,1]
    else:
        seq = sequence
        outseq = outsequence
        
    encoder = get_encoder(model, input_1, encoded)
    decoder = get_decoder(model, ENCODING_N_DEFAULT)
    
    encodings = encoder(seq)
    repeater = RepeatVector(outseq.shape[1])(encodings)
    merged = concatenate([repeater, outseq], axis=-1)
    decodings = decoder(merged)

    #decodings = model([seq, outseq])
    prep = np.load(prep_fn)
    bandmin = prep['bandmin']
    bandmax = prep['bandmax']
        
    # rescale back to original ABSOLUTE mags
    decodings = decodings * ( bandmin - bandmax) - bandmin
    
    if sn_name is None:
        return decodings
    
    # convert back to apparent magnitues
    lcs = np.load(lc_file, allow_pickle=True)['lcs']
    
    for lc in lcs:
        if lc.name == sn_name:
            redshift = lc.redshift
            k_correction = 2.5 * np.log10(1.+redshift)
            dist = cosmo.luminosity_distance([redshift]).value[0]  # returns dist in Mpc
            fluxes = 10**((decodings - 26.3 - k_correction + 5. * np.log10(dist*1e5))/(-2.5))
            return outseq[0,:,0], fluxes[0].numpy()
    
    
    
    
    
    
    