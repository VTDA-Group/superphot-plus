from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from torch.utils.data import TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from astropy.cosmology import Planck13 as cosmo

from superphot_plus.posterior_samples import PosteriorSamples
from superphot_plus.supernova_class import SupernovaClass as SnClass

@dataclass
class PosteriorSamplesGroup:
    """Holds data from multiple objects' posterior objects."""

    posterior_objects: List[PosteriorSamples]
    use_redshift_info: Optional[bool] = False
    ignore_param_idxs: Optional[List[int]] = field(default_factory=list)
    random_seed: Optional[int] = None
    
    def __post_init__(self):
        self.names = np.array([
            ps.name for ps in self.posterior_objects
        ])
        self.labels = np.array([
            ps.sn_class for ps in self.posterior_objects
        ])
        self.redshifts = np.array([
            ps.redshift for ps in self.posterior_objects
        ])
        self.abs_mags = []
        self.rng = np.random.default_rng(
            self.random_seed
        )

        # save equal number of draws per LC
        num_samples = [ps.samples.shape[0] for ps in self.posterior_objects]
        self.num_draws = min(num_samples)
        
        feat_arr = []
        median_feats = []
        
        for ps in self.posterior_objects:
            samples = ps.samples[:self.num_draws]
            
            if self.use_redshift_info:
                z_arr = np.ones((self.num_draws, 2)) * ps.redshift
                max_flux = ps.max_flux
                if max_flux is None:
                    self.abs_mags.append(None)
                    z_arr[:,1] = -np.inf
                else:
                    k_corr = 2.5 * np.log10(1.+ps.redshift)
                    dist = cosmo.luminosity_distance([ps.redshift]).value[0]  # returns dist in Mpc
                    abs_mag = -2.5 * np.log10(max_flux) + 26.3 - 5. * np.log10(dist*1e5) + k_corr
                    self.abs_mags.append(abs_mag)
                    z_arr[:,1] = abs_mag

                samples = np.append(
                    samples, z_arr, axis=1
                )
            samples = np.delete(samples, self.ignore_param_idxs, 1)
            feat_arr.extend(samples)
            median_feats.append(np.median(samples, axis=0))
            
        self.features = np.asarray(feat_arr)
        self.median_features = np.asarray(median_feats)
        

    def __iter__(self):
        return iter((self.names, self.labels, self.redshifts))

    
    def oversample(self, fits_per_majority_lc=1):
        """Oversamples, drawing from posteriors of a certain fit.
        Assumes goal_per_class is the number of majority class if not set.
        
        Returns
        -------
        tuple of np.ndarray
            Tuple containing oversampled features and labels.
        """

        oversampled_labels = []
        oversampled_features = []
        labels_unique, counts = np.unique(
            self.labels, return_counts=True
        )
        
        goal_per_class = np.max(counts) * fits_per_majority_lc
        
        for l in labels_unique:
            idxs_in_class = np.asarray(self.labels == l).nonzero()[0]
            samples_per_fit = max(round(goal_per_class / len(idxs_in_class)), 1)

            for i in idxs_in_class:
                sampled_idx = self.rng.choice(
                    np.arange(self.num_draws),
                    samples_per_fit
                )
                sampled_features = self.features[i*self.num_draws + sampled_idx]
                
                oversampled_features.extend(list(sampled_features))
                oversampled_labels.extend([l] * samples_per_fit)
                
        return np.array(oversampled_features), np.array(oversampled_labels)
    
    
    def oversample_smote(self):
        """
        Uses SMOTE to oversample data from rarer classes.
        """
        oversample = SMOTE()
        features_smote, labels_smote = oversample.fit_resample(
            self.median_features,
            self.labels
        )
        return features_smote, labels_smote
    
    
    def split(self, split_frac=0.1, split_indices=None, shuffle=True):
        
        if split_indices is not None:
            idx1, idx2 = split_indices
        else:
            idx1, idx2 = train_test_split(
                np.arange(len(self.labels)),
                stratify=self.labels,
                test_size=split_frac,
                random_state=self.random_seed
            )
            
        split_1 = PosteriorSamplesGroup(
            self.posterior_objects[idx1],
            self.use_redshift_info,
            self.ignore_param_idxs,
            self.random_seed
        )
        split_2 = PosteriorSamplesGroup(
            self.posterior_objects[idx2],
            self.use_redshift_info,
            self.ignore_param_idxs,
            self.random_seed
        )
        
        return split_1, split_2
    
    def canonicalize_labels(self):
        """Convert labels to canon labels.
        """
        self.labels = np.asarray([
            SnClass.canonicalize(l) for l in self.labels
        ])
        
    def make_binary(self, target_label="SN Ia"):
        """Convert labels to a binary classification
        problem."""
        self.labels = np.where(
            self.labels == target_label,
            target_label,
            "other"
        )
        
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
        return None
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
        

@dataclass
class TrainData:
    """Holds train and validation datasets."""

    train_dataset: TensorDataset
    val_dataset: TensorDataset
    
    def __iter__(self):
        return iter((self.train_dataset, self.val_dataset))


@dataclass
class TestData:
    """Holds information about testing data."""

    test_features: np.ndarray
    test_classes: np.ndarray
    test_names: np.ndarray
    
    def __post_init__(self):
        """Ensure everything is numpy arrays."""
        self.test_features = np.asarray(self.test_features)
        self.test_classes = np.asarray(self.test_classes)
        self.test_names = np.asarray(self.test_names)

    def __iter__(self):
        return iter(
            (
                self.test_features,
                self.test_classes,
                self.test_names,
            )
        )
