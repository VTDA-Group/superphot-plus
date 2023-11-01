"""Run SuperRAENN on superphot+ light curves."""
import pandas as pd
import os
import numpy as np
from astropy.table import Table
from astropy.cosmology import Planck13 as cosmo

from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import convert_mags_to_flux
from superphot_plus.lightcurve import Lightcurve

from superraenn.preprocess import save_lcs
from superraenn.lc import LightCurve
from superraenn.raenn import *
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input


def prep_lcs_superraenn(dataset_csv, data_dir, save_dir, metafile):
    """Run equivalent of superraenn-prep on processed light curves."""
    print("STARTS")
    os.makedirs(save_dir, exist_ok=True)
    
    full_df = pd.read_csv(dataset_csv)
    all_names = full_df.NAME.to_numpy()
    labels = full_df.CLASS.to_numpy()
    redshifts = full_df.Z.to_numpy()
        
    my_lcs = []
    
    with open(metafile, 'w+') as f:
        f.write('# SN Redshift Type T_explosion MW(EBV)')

    for i, name in enumerate(all_names):
        if i % 500 == 0:
            print(i)
            
        if (np.isnan(redshifts[i]) or redshifts[i] <= 0):
            continue
        
        l_canon = SnClass.canonicalize(labels[i])
            
        lc = Lightcurve.from_file(
            os.path.join(
                data_dir,
                name + ".npz"
            )
        )
        
        # lc is already extinction-subtracted, phase-shifted
        
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
            print(np.mean(my_lc.dense_lc[:30,0,0] - my_lc.dense_lc[:30,1,0]))
            my_lcs.append(my_lc)
            
            with open(metafile, 'a') as f:
                f.write(f'{name} {redshifts[i]} {l_canon} 0.0 0.0')
                
        except:
            print("skipped")
            continue
        
    save_lcs(my_lcs, save_dir)
    
    
    
def run_superraenn_raenn():
    lcfile = "superraenn_outputs/lcs.npz"
    outdir = "superraenn_outputs"
    
    NEURON_N_DEFAULT = 100
    ENCODING_N_DEFAULT = 10
    N_EPOCH_DEFAULT = 1000

    sequence, outseq, ids, maxlen, nfilts = prep_input(lcfile, save=True, outdir=outdir)

    model, callbacks_list, input_1, encoded = make_model(
        NEURON_N_DEFAULT,
        ENCODING_N_DEFAULT,
        int(maxlen),
        2
    )
    
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

    
def retrieve_decodings(sn_name=None):
    model_path = "superraenn_outputs/models/model.h5"
    lcfile = "superraenn_outputs/lcs.npz"
    
    model = load_model(
        model_path,
        custom_objects={"customLoss": customLoss}
    )
    input_1 = Input((None, 5))
    mid_layer = model.layers[1](input_1)
    encoded = model.layers[2](mid_layer)
    
    sequence, outsequence, ids, maxlen, nfilts = prep_input(
        lcfile, save=False,
    )
    
    ids = np.array(ids)
    
    if sn_name is not None:
        seq = sequence[ids == sn_name]
        #outseq = outsequence[ids == sn_name]
        outseq_single = outsequence[ids == sn_name]
        outseq = np.zeros((1, 1000, 2))
        outseq[0, :, 0] = np.linspace(
            -30,
            100,
            num=1000
        )
        outseq[0, :, 1] = outseq_single[0,0,1]
    else:
        seq = sequence
        outseq = outsequence
        
    print(outseq)

    encoder = get_encoder(model, input_1, encoded)
    decoder = get_decoder(model, 10)
    
    decodings = get_decodings(
        decoder, encoder,
        seq, outseq, 10,
        plot=True
    )
    
    prep = np.load("superraenn_outputs/prep.npz")
    bandmin = prep['bandmin']
    bandmax = prep['bandmax']
        
    # rescale back to original ABSOLUTE mags
    decodings = decodings * ( bandmin - bandmax) - bandmin
    
    if sn_name is None:
        return decodings
    
    # convert back to apparent magnitues
    lcs = np.load(lcfile, allow_pickle=True)['lcs']
    
    for lc in lcs:
        if lc.name == sn_name:
            redshift = lc.redshift
            k_correction = 2.5 * np.log10(1.+redshift)
            dist = cosmo.luminosity_distance([redshift]).value[0]  # returns dist in Mpc
            fluxes = 10**((decodings - 26.3 - k_correction + 5. * np.log10(dist*1e5))/(-2.5))
            return outseq[0,:,0], fluxes[0]
    
    
    
    
    
    
    