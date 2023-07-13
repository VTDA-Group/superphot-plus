"""
This package contains the dust maps for superphot_plus.
"""
import os

import superphot_plus

__all__ = ["dust_filepath"]

dust_filepath = os.path.dirname(superphot_plus.__file__)
