# Using SNAPI, imports ZTF data directly from TNS and ALeRCE, including non-detections.
import os
import numpy as np
import itertools
import multiprocessing
from functools import partial

from astropy.cosmology import Planck15  # pylint: disable=no-name-in-module
from snapi import Transient, Photometry, TransientGroup
from snapi.query_agents import TNSQueryAgent, ALeRCEQueryAgent


def single_worker_import(batch, skipped_names_fn):
    """Single worker's script to run in parallel."""
    name_batch, tns_agent, alerce_agent = batch
    single_name_import_static = partial(
        single_name_import,
        tns_agent=tns_agent,
        alerce_agent=alerce_agent,
        skipped_names_fn=skipped_names_fn
    )
    
    transients = []
    for n in name_batch:
        transients.append(
            single_name_import_static(n)
        )
    return transients


def single_name_import(
    n, tns_agent, alerce_agent,
    skipped_names_fn
):
    transient = Transient(iid=n)
    qr_tns, success = tns_agent.query_transient(transient, local=True) # we dont want spectra
    if not success:
        with open(skipped_names_fn, "a") as f:
            f.write(f"{n}: TNS query failed\n")
        return
    for result in qr_tns:
        transient.ingest_query_info(result.to_dict())

    ztf_name = None
    for n in transient.internal_names:
        if n[:3] == "ZTF":
            ztf_name = n

    if ztf_name is None:
        with open(skipped_names_fn, "a") as f:
            f.write(f"{n}: No ZTF\n")
        return
    
    qr_alerce, success = alerce_agent.query_transient(transient)
    if not success:
        with open(skipped_names_fn, "a") as f:
            f.write(f"{n}: ALeRCE query failed\n")
        return
    for result in qr_alerce:
        transient.ingest_query_info(result.to_dict())

    # quality cuts
    if transient.photometry is None:
        with open(skipped_names_fn, "a") as f:
            f.write(f"{n}: No photometry.\n")
        return

    phot = transient.photometry
    phot.filter_subset(["ZTF_r", "ZTF_g"], inplace=True)

    if len(phot.detections['filter'].unique()) < 2:
        with open(skipped_names_fn, "a") as f:
            f.write(f"{n}: Data in fewer than two filters.\n")
        return
    
    phot.phase(inplace=True)
    phot.truncate(min_t=-50., max_t=100.)
    phot.correct_extinction(coordinates=transient.coordinates, inplace=True)
    try:
        peak_idx = (phot.detections['flux'] - phot.detections['flux_error']).dropna().idxmax()
        transient.max_flux = np.max(phot.detections.loc[peak_idx, 'flux'])
        phot.normalize(inplace=True)
    except:
        print(phot.detections)
        return
    
    transient.peak_abs_mag = calculate_absolute_magnitude(transient)
    transient.meta_attrs.extend(['max_flux', 'peak_abs_mag'])

    if len(phot.detections['filter'].unique()) < 2:
        with open(skipped_names_fn, "a") as f:
            f.write(f"{n}: Data in fewer than two filters.\n")
        return

    high_snr_detections = phot.detections.loc[
        phot.detections['mag_error'] <= (5 / 6. / np.log(10))
    ]

    for b in ['ZTF_r', 'ZTF_g']:
        # SNR >= 3
        high_snr_b = high_snr_detections.loc[high_snr_detections['filter'] == b]
        # number of high-SNR detections cut
        if len(high_snr_b) < 5:
            with open(skipped_names_fn, "a") as f:
                f.write(f"{n}: Not enough high-SNR detections\n")
            return

        # variability cut
        if np.ptp(high_snr_b['mag']) < 3 * high_snr_b['mag_error'].mean():
            with open(skipped_names_fn, "a") as f:
                f.write(f"{n}: Amplitude too small\n")
            return

        # second variability cut
        if high_snr_b['mag'].std() < high_snr_b['mag_error'].mean():
            with open(skipped_names_fn, "a") as f:
                f.write(f"{n}: Variability too small\n")
            return
    
    if ~np.isnan(transient.redshift) and (transient.redshift > 0):
        phot.times /= (1. + transient.redshift)

    transient.photometry = phot
    return transient
    
    
def import_all_names(
    names,
    save_dir,
    max_n = 100_000,
    checkpoint_freq = None,
    n_cores: int = 1,
    overwrite: bool = False
): # pylint: disable=invalid-name
    """Extract all spectroscopic SNe II from TNS and save with SNAPI.

    Parameters
    ----------
    save_dir : str
        Directory to save extracted data.
    """
    pool = multiprocessing.Pool(n_cores)
    
    # make file for skipped names
    skipped_names_fn ="skipped_names.txt"
    skipped_names = []
    if (not overwrite) and (os.path.exists(skipped_names_fn)):
        with open(skipped_names_fn, "r") as f:
            for row in f:
                skipped_names.append(row.split(":")[0])
        
        if os.path.exists(save_dir):
            tg = TransientGroup.load(save_dir)
            print(f"{len(tg.metadata.index)} events already saved.")
            skipped_names.extend(list(tg.metadata.index))
            transients = [t for t in tg if t.id in names]
    else:
        with open(skipped_names_fn, "w") as f:
            f.write("")
        transients = []
            
    names_keep = [n for n in names if n not in skipped_names][:max_n]
    
    single_worker_import_static = partial(
        single_worker_import,
        skipped_names_fn=skipped_names_fn,
    )
    
    print(f"{len(names_keep)} names to query across {n_cores} cores.")
    
    if checkpoint_freq is not None:
        num_checkpoints = len(names_keep) // checkpoint_freq
        checkpoint_batches = [names_keep[i::num_checkpoints] for i in range(num_checkpoints)]
        tns_agents = [TNSQueryAgent() for _ in range(num_checkpoints)]
        alerce_agents = [ALeRCEQueryAgent() for _ in range(num_checkpoints)]

        for _, cb in enumerate(checkpoint_batches):
            name_batches = [cb[i::n_cores] for i in range(n_cores)]
            print(f"Processing {len(cb)} transients in batch")
            result = pool.map(single_worker_import_static, zip(name_batches, tns_agents, alerce_agents))
            transients_loop = list(itertools.chain(*result))
            print("Finished processing, making transient group now")
            transients.extend(filter(None, transients_loop))
            transient_group = TransientGroup(transients)
            transient_group.save(save_dir)
            print(f"Total transients saved: {len(transient_group)}.")
            
    else:
        names_batches = [names_keep[i::n_cores] for i in range(n_cores)]
        tns_agents = [TNSQueryAgent() for _ in range(n_cores)]
        alerce_agents = [ALeRCEQueryAgent() for _ in range(n_cores)]
        transients.extend(
            pool.map(single_worker_import_static, zip(names_batches, tns_agents, alerce_agents))
        )
        transient_group = TransientGroup(filter(None, transients))
        transient_group.save(save_dir)


def calculate_absolute_magnitude(transient):
    k_corr = 2.5 * np.log10(1.0 + transient.redshift)
    distmod = Planck15.distmod(transient.redshift).value
    peak_mag = -2.5 * np.log10(transient.max_flux) + 23.9 - distmod + k_corr
    print(peak_mag)
    return peak_mag
        
        