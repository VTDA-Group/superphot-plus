from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from torch.utils.data import TensorDataset
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

from superphot_plus.posterior_samples import PosteriorSamples


@dataclass
class PosteriorSamplesGroup:
    """Holds data from multiple objects' posterior objects."""

    posterior_objects: List[PosteriorSamples]
    use_redshift_info: Optional[bool] = False
    ignore_param_idxs: Optional[List[int]] = field(default_factory=list)

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

        # save equal number of draws per LC
        num_samples = [ps.samples.shape[0] for ps in self.posterior_objects]
        self.num_draws = min(num_samples)
        feat_arr = []
        
        for ps in self.posterior_objects:
            samples = ps.samples[:self.num_draws]
            
            if self.use_redshift_info:
                z_arr = np.ones((self.num_draws, 2)) * ps.redshift
                max_flux = ps.max_flux
                if max_flux is None:
                    self.abs_mags.append(None)
                    z_arr[:,1] = -np.inf
                else:
                    self.abs_mags.append(np.log10(max_flux))
                    z_arr[:,1] = np.log10(max_flux)

                samples = np.append(
                    samples, z_arr, axis=1
                )
            samples = np.delete(samples, self.ignore_param_idxs, 1)
            feat_arr.extend(samples)
            
        self.features = np.asarray(feat_arr)
        self.median_features = np.median(feat_arr, axis=0)
        

    def __iter__(self):
        return iter((self.names, self.labels, self.redshifts))

    
    def oversample(self, goal_per_class):
        """Oversamples, drawing from posteriors of a certain fit.
        
        goal_per_class : int
            Number of samples per class.

        Returns
        -------
        tuple of np.ndarray
            Tuple containing oversampled features and labels.
        """

        oversampled_labels = []
        oversampled_features = []
        labels_unique = np.unique(self.labels)

        for l in labels_unique:
            idxs_in_class = np.asarray(self.labels == l).nonzero()[0]
            samples_per_fit = round(goal_per_class / len(idxs_in_class))

            for i in idxs_in_class:
                sampled_idx = np.random.choice(
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
                test_size=0.1
            )
            
        split_1 = PosteriorSamplesGroup(
            self.posterior_objects[idx1],
            self.use_redshift_info,
            self.ignore_param_idxs
        )
        split_2 = PosteriorSamplesGroup(
            self.posterior_objects[idx2],
            self.use_redshift_info,
            self.ignore_param_idxs
        )
        return split_1, split_2


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
