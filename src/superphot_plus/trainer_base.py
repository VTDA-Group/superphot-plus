import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from astropy.cosmology import Planck13 as cosmo
from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.format_data_ztf import import_labels_only, oversample_using_posteriors
from superphot_plus.model.data import ZtfData
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.utils import adjust_log_dists


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
        self.allowed_types = ["SN Ia", "SN II", "SN IIn", "SLSN-I", "SN Ibc"]

        # Fitting method
        self.sampler = sampler
        self.include_redshift = include_redshift
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
        names, labels, redshifts = import_labels_only(
            input_csvs=input_csvs,
            allowed_types=self.allowed_types,
            fits_dir=self.fits_dir,
            sampler=self.sampler,
        )
        
        if self.include_redshift:
            skip_idxs = ((np.isnan(redshifts)) | (redshifts <= 0))
            names = names[~skip_idxs]
            labels = labels[~skip_idxs]
            redshifts = redshifts[~skip_idxs]
            
        for (train_index, test_index) in kf.split(names, labels):
            train_names = names[train_index]
            train_labels = labels[train_index]
            train_redshifts = redshifts[train_index]
            
            test_names = names[test_index]
            test_labels = labels[test_index]
            test_redshifts = redshifts[test_index]
            
            train_data = ZtfData(train_names, train_labels, train_redshifts)
            test_data = ZtfData(test_names, test_labels, test_redshifts)
            
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
        
        if self.include_redshift:
            skip_idxs = ((np.isnan(redshifts)) | (redshifts <= 0))
            names = names[~skip_idxs]
            labels = labels[~skip_idxs]
            redshifts = redshifts[~skip_idxs]
        
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
            
        names, test_names, labels, test_labels, redshifts, test_redshifts = train_test_split(
            names, labels, redshifts, stratify=labels, shuffle=True, test_size=0.1
        )
        train_data = ZtfData(names, labels, redshifts)
        test_data = ZtfData(test_names, test_labels, test_redshifts)

        return train_data, test_data

    def generate_train_data(self, train_data, goal_per_class, train_index, val_index):
        """Extracts and processes the data for training and validation.
        Oversamples the features to tackle the supernovae class imbalance
        and adjusts them to their log distributions.

        Parameters
        ----------
        train_data : ZtfData
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
        names, labels, redshifts = train_data

        train_names, val_names = names[train_index], names[val_index]
        train_labels, val_labels = labels[train_index], labels[val_index]

        # Convert labels to classes
        train_classes = SnClass.get_classes_from_labels(train_labels)
        val_classes = SnClass.get_classes_from_labels(val_labels)

        train_features, train_classes = oversample_using_posteriors(
            lc_names=train_names,
            labels=train_classes,
            goal_per_class=goal_per_class,
            fits_dir=self.fits_dir,
            sampler=self.sampler,
            oversample_redshifts=self.include_redshift,
            redshifts=redshifts,
            chisq_cutoff=1.2,
        )
        val_features, val_classes = oversample_using_posteriors(
            lc_names=val_names,
            labels=val_classes,
            goal_per_class=round(0.1 * goal_per_class),
            fits_dir=self.fits_dir,
            sampler=self.sampler,
            oversample_redshifts=self.include_redshift,
            redshifts=redshifts,
            chisq_cutoff=1.2,
        )

        train_features = adjust_log_dists(train_features, redshift=self.include_redshift)
        val_features = adjust_log_dists(val_features, redshift=self.include_redshift)
        
        return train_features, train_classes, val_features, val_classes

    
    def generate_test_data(self, test_data: ZtfData):
        """Extracts and processes the data for testing, adjusting the
        features to their log distributions.

        Parameters
        ----------
        test_data : ZtfData
            Contains the ZTF object names, classes and redshifts for testing.

        Returns
        -------
        tuple
            A tuple containing the final test features and respective classes,
            the corresponding test ZTF object names and test group indices.
        """
        test_names, test_labels, test_redshifts = test_data

        test_features = []
        test_classes_os = []
        test_group_idxs = []
        test_names_os = []

        test_classes = SnClass.get_classes_from_labels(test_labels)

        for i, test_name in enumerate(test_names):
            test_posts, kwargs = get_posterior_samples(test_name, self.fits_dir, self.sampler)
            if np.mean(test_posts[:,-1]) > 1.2:
                continue
            test_classes_os.extend([test_classes[i]] * len(test_posts))
            test_names_os.extend([test_names[i]] * len(test_posts))
            if self.include_redshift:
                z_ext = np.ones((len(test_posts), 2)) * test_redshifts[i]
                k_correction = 2.5 * np.log10(1.+test_redshifts[i])
                dist = cosmo.luminosity_distance([test_redshifts[i],]).value  # returns dist in Mpc
                z_ext[:,1] = -2.5*np.log10(kwargs['max_flux']) + 26.3 \
                    - 5. * np.log10(dist*1e6/10.0) + k_correction
                
                test_posts = np.append(test_posts, z_ext, axis=1)
            if len(test_group_idxs) == 0:
                start_idx = 0
            else:
                start_idx = test_group_idxs[-1][-1] + 1
                
            test_features.extend(list(test_posts))
            test_group_idxs.append(np.arange(start_idx, start_idx + len(test_posts)))

        test_features = np.array(test_features)
        test_classes = np.array(test_classes_os)
        test_names = np.array(test_names_os)

        test_features = adjust_log_dists(test_features, redshift=self.include_redshift)

        return test_features, test_classes, test_names, test_group_idxs
