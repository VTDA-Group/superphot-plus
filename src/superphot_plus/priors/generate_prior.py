import os

import pandas as pd

from .superphot_prior import SuperphotPrior

def generate_priors(filts: list[str], reference_band: str='ZTF_r'):
    """Generate SamplerPrior for Superphot+ samplers, given a list of
    SNAPI filters and a reference band. Assumes relative priors.
    """
    concat_df = None
    for f in filts:
        prior_df = pd.read_csv(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
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
    