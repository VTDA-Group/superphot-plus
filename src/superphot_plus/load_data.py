import os
import numpy as np

from superphot_plus.file_paths import INPUT_CSVS
from superphot_plus.file_utils import get_multiple_posterior_samples
from superphot_plus.format_data_ztf import import_labels_only
from superphot_plus.supernova_properties import SupernovaProperties
from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.supernova_class import SupernovaClass as SnClass


def read_classification_data(sampler, allowed_types, fits_dir, input_csvs=None):
    """Imports classification data from disk.

    Parameters
    ----------
    sampler : str
        Method to use for fitting lightcurves.
    allowed_types : list of str
        List of allowed types for labels.
    fits_dir : str
        Output directory path.
    input_csvs : list of str
        The list of training CSV files. Defaults to None.

    Returns
    -------
    tuple
        The names, the labels, the redshifts, and a dictionary
        containing the posterior samples for each light curve.
    """
    if input_csvs is None:
        input_csvs = INPUT_CSVS

    names, labels, redshifts = import_labels_only(
        input_csvs=input_csvs,
        allowed_types=allowed_types,
        fits_dir=fits_dir,
        sampler=sampler,
    )

    posterior_samples = get_multiple_posterior_samples(names, fits_dir, sampler)

    return (
        np.array(names),
        np.array(labels),
        np.array(redshifts),
        posterior_samples,  # Dict of light curve posteriors
    )


def read_mosfit_data(sampler, params_dir, fits_dir):
    """Imports posteriors and supernova physical properties
    for regression, for all the light curves stored on disk.

    Parameters
    ----------
    sampler : str
        Method used for fitting the lightcurves.
    params_dir : list of str
        Directory where physical property values are stored.
    fits_dir : str
        Directory where posterior samples are stored.

    Returns
    -------
    tuple of np.array
        The names, the posterior samples and the physical
        properties for each light curve.
    """
    names = []
    posteriors = []
    properties = []

    for file in os.listdir(params_dir):
        filename = file.split(".")[0]
        names.append(filename)

        posts = PosteriorSamples.from_file(
            input_dir=fits_dir,
            name=filename,
            sampling_method=sampler,
            sn_class=SnClass.SUPERLUMINOUS_SUPERNOVA_I,
        )
        posteriors.append(posts.sample_mean())

        props = SupernovaProperties.from_file(
            input_dir=params_dir,
            name=filename,
        )
        properties.append(props)

    return (
        np.array(names),
        np.array(posteriors),
        np.array(properties),
    )
