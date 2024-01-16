import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from astropy.cosmology import Planck13 as cosmo
from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.format_data_ztf import import_labels_only
from superphot_plus.model.data import PosteriorSamplesGroup
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.format_data_ztf import retrieve_posterior_set

class TrainerBase:
    """Trainer base class."""

    def __init__(
        self,
        config_name,
        fits_dir, #TODO: make optional from config
        sampler="dynesty",
        model_type='LightGBM',
        include_redshift=True,
        probs_file=None,
        n_folds=10,
    ):
        # Supernova class types
        # TODO: replace with supernova_class enumeration
        self.allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]

        # Fitting method
        self.sampler = sampler
        self.include_redshift = include_redshift
        if self.include_redshift:
            self.skipped_params = [3,]
        else:
            self.skipped_params = [0,3]
        self.probs_file = probs_file
        self.fits_dir = fits_dir
        
        self.config_name = config_name
        self.models, self.configs = [], []
        self.model_type = model_type
        
        # generate k-folds
        self.n_folds = max(int(n_folds), 1)
        self.random_seed = 1 # TODO: un-hard code this
        
        if self.n_folds > 1:
            self.kf = StratifiedKFold(self.n_folds, random_state=self.random_seed, shuffle=True)
        else:
            self.kf = None

    
    def k_fold_split_train_test(self, kf, input_csvs=None, rng_seed=None):
        """Reads data and splits into n K-folds. Outputs n sets
        of train/test sets.
        
        Parameters
        ----------
        input_csvs : list of str
            List of input CSV file paths.

        Returns
        -------
        list of 2-tuples
            N sets of the train data and the test data.
        """
        k_fold_datasets = []
        
        if input_csvs is None:
            input_csvs = INPUT_CSVS

        # Load train and test data (holdout of 10%)
        names, labels, redshifts = self.load_csv(
            input_csvs=input_csvs,
        )
        
        all_post_objs = retrieve_posterior_set(
            names, self.fits_dir, sampler=self.sampler,
            redshifts=redshifts,
            labels=labels,
            chisq_cutoff=1.2
        )

        for (train_index, test_index) in kf.split(names, labels):
            train_posteriors = all_post_objs[train_index]
            test_posteriors = all_post_objs[test_index]
            
            train_data = PosteriorSamplesGroup(
                train_posteriors,
                use_redshift_info=self.include_redshift,
                ignore_param_idxs=self.skipped_params
            )
            test_data = PosteriorSamplesGroup(
                test_posteriors,
                use_redshift_info=self.include_redshift,
                ignore_param_idxs=self.skipped_params
            )
            
            k_fold_datasets.append((train_data, test_data))
            
        return k_fold_datasets
    
    def load_csv(self, input_csvs):
        """Load CSV data.
        """
        # Load train and test data (holdout of 10%)
        names, labels, redshifts = import_labels_only(
            input_csvs=input_csvs,
            allowed_types=self.allowed_types,
            fits_dir=self.fits_dir,
            sampler=self.sampler,
        )
        
        if not self.include_redshift:
            redshifts = np.ones(len(names)) # just set all to valid z's
        
        return names, labels, redshifts
        
    def split_train_test(self, input_csvs=None):
        """Reads data and splits it into training and testing sets.

        Parameters
        ----------
        input_csvs : list of str
            List of input CSV file paths.

        Returns
        -------
        tuple
            The train data and the test data.
        """
        if input_csvs is None:
            input_csvs = INPUT_CSVS

        names, labels, redshifts = self.load_csv(input_csvs)
        all_post_objs = retrieve_posterior_set(
            names, self.fits_dir, sampler=self.sampler,
            labels=labels,
            redshifts=redshifts,
            chisq_cutoff=1.2
        )
        train_idxs, test_idxs = train_test_split(
            np.arange(len(labels)),
            stratify=labels,
            shuffle=True,
            test_size=0.1
        )
        train_data = PosteriorSamplesGroup(
            all_post_objs[train_idxs],
            use_redshift_info=self.include_redshift,
            ignore_param_idxs=self.skipped_params
        )
        test_data = PosteriorSamplesGroup(
            all_post_objs[test_idxs],
            use_redshift_info=self.include_redshift,
            ignore_param_idxs=self.skipped_params
        )

        return train_data, test_data

    
    def generate_train_data(self, train_data, goal_per_class, train_index, val_index):
        """Extracts and processes the data for training and validation.
        Oversamples the features to tackle the supernovae class imbalance
        and adjusts them to their log distributions.

        Parameters
        ----------
        train_data : PosteriorSamplesGroup
            Contains the ZTF object names, classes and redshifts for training.
        goal_per_class : int
            The number of samples for each supernova class (for oversampling).
        train_index : np.ndarray
            The indices for the training data samples.
        val_index : np.ndarray
            The indices for the validation data samples.

        Returns
        -------
        tuple
            A tuple containing the final training features and respective classes,
            and validation features and respective classes.
        """
        train_data, val_data = train_data.split(split_frac=0.1)
        
        train_features, train_labels = train_data.oversample(
            goal_per_class=round(0.9*goal_per_class),
        )
        val_features, val_labels = train_data.oversample(
            goal_per_class=round(0.1*goal_per_class),
        )

        train_classes = SnClass.get_classes_from_labels(train_labels)
        val_classes = SnClass.get_classes_from_labels(val_labels)
        
        return train_features, train_classes, val_features, val_classes

    
    def generate_test_data(self, test_data: PosteriorSamplesGroup):
        """Extracts and processes the data for testing, adjusting the
        features to their log distributions.

        Parameters
        ----------
        test_data : PosteriorSamplesGroup
            Contains the ZTF object names, classes and redshifts for testing.

        Returns
        -------
        tuple
            A tuple containing the final test features and respective classes,
            the corresponding test ZTF object names and test group indices.
        """
        test_names, test_labels, test_redshifts = test_data
        test_classes = SnClass.get_classes_from_labels(test_labels)
        test_features = test_data.features
        os_classes = np.flatten(
            [[c] * test_data.num_draws for c in test_classes]
        )
        os_names = np.flatten(
            [[n] * test_data.num_draws for n in test_names]
        )
        return test_features, os_classes, os_names
