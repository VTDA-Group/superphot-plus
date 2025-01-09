import os
from typing import Optional

import pandas as pd

from .superphot_prior import SuperphotPrior

def generate_priors(filts: list[str], reference_band: str='ZTF_r', priors_dir: Optional[str]=None):
    """Generate SamplerPrior for Superphot+ samplers, given a list of
    SNAPI filters and a reference band. Assumes relative priors.
    """
    if priors_dir is None:
        priors_dir = os.path.dirname(os.path.realpath(__file__))
    concat_df = None

    for f in filts:
        prior_df = pd.read_csv(
            os.path.join(
                priors_dir,
                f'priors_{f}.csv'
            )
        )
        if f != reference_band:
            prior_df['relative'] = [f'{p}_{reference_band}' for p in prior_df['param']]
        prior_df['param'] = [f'{p}_{f}' for p in prior_df['param']]
        if concat_df is None:
            concat_df = prior_df
        else:
            concat_df = pd.concat([concat_df, prior_df], ignore_index=True)
        
    return SuperphotPrior(concat_df)
    