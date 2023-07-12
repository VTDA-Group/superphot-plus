"""
This package contains the dust maps for superphot_plus.
"""
import os
from pathlib import Path

from astropy.utils.data import get_pkg_data_filename

import superphot_plus

__all__ = ["rootdir", "get_dust_filepath"]

rootdir = Path(os.path.dirname(superphot_plus.__file__)) / "sfd"


def get_dust_filepath(**kwargs):
    """Return the path to the directory with the dust files.

    Returns
    -------
    filepath : `str`
        The path of the directory.
    """
    return get_pkg_data_filename(filename, package="superphot_plus.sfd", **kwargs)
