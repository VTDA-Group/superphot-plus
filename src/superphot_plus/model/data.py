from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from torch.utils.data import TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from astropy.cosmology import Planck13 as cosmo
from snapi.analysis import SamplerResult

from superphot_plus.supernova_class import SupernovaClass as SnClass

        
        
    def __iter__(self):
        return iter((self.names, self.labels, self.redshifts))

        
    def make_fully_redshift_independent(self):
        """Experimental!
        We can convert our shape parameters to be FULLY
        z-independent by instead using:
        tau_rise/gamma, tau_rise/tau_fall, beta*tau_rise
        
        (but its log scale for tau_rise, gamma, tau_fall
        so add/subtract instead)
        
        We do everything relative to tau_rise because that's
        the first shape param to be measured in real time!
        """
        self.features_z_independent = np.asarray([
            np.log10(self.features[:,0]) + self.features[:,2],
            self.features[:,2] - self.features[:,1],
            self.features[:,2] - self.features[:,3],
        ]).T
        self.features_z_independent = np.append(
            self.features_z_independent,
            self.features[:,4:],
            axis=1
        )
        self.features = self.features_z_independent

