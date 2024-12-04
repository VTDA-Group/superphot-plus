import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from astropy.cosmology import Planck13 as cosmo
from snapi import TransientGroup, SamplerResultGroup

from .config import SuperphotConfig
from .supernova_class import SupernovaClass as SnClass

class TrainerBase:
    """Trainer base class."""

    def __init__(
        self,
        config: SuperphotConfig
    ):
        self.config = config
        self.models = []
        if self.config.n_folds > 1:
            self.kf = StratifiedKFold(
                self.config.n_folds,
                random_state=self.config.random_seed,
                shuffle=True
            )
        else:
            self.kf = None

    def retrieve_transient_metadata(self, transient_group: TransientGroup):
        """Filter transient group info and return relevant metadata."""
        if self.config.target_label is not None:
            transient_group.add_binary_class(self.config.target_label)
            label_name = f'binary_class_{self.config.target_label}'
        else:
            transient_group.canonicalize_classes(SnClass.canonicalize)
            label_name = 'canonical_class'
        
        if self.config.use_redshift_features:
            transient_group.add_col('abs_mag', lambda x: x.photometry.absolute(x.redshift).detections['mag'].dropna().min())
            metadata = transient_group.metadata.loc[:,['abs_mag','redshift',label_name]]
        else:
            metadata = transient_group.metadata.loc[:,[label_name,]]
        
        metadata.dropna(inplace=True)
        metadata = metadata[metadata[label_name].isin(self.config.allowed_types)]
            
        if self.config.target_label is None:
            metadata['label'] = SnClass.get_classes_from_labels(metadata['canonical_class'])
        else:
            metadata['label'] = metadata[f'binary_class_{self.config.target_label}'].astype(int) # 1 if label, 0 otherwise
        
        return metadata
            
    def retrieve_sampler_results(self, srg: SamplerResultGroup, metadata: pd.DataFrame, balance_classes=False):
        """From transient group info, retrieve dataframe
        containing all sampling posterior info.
                """
        if balance_classes:
            class_dict = {x.Index: x.label for x in metadata.itertuples()}
            srg.balance_classes(class_dict, self.config.fits_per_majority)
            
        all_samples = srg.all_samples
        filt_samples = all_samples.loc[
            (all_samples['score'] <= self.config.chisq_cutoff) & (
                all_samples['sampler'] == self.config.sampler
            )
        ]
        
        if self.config.use_redshift_features:
            df = pd.merge(
                filt_samples,
                metadata.loc[:,['label', 'redshift', 'abs_mag']],
                how='inner',
                left_index=True,
                right_index=True
            )
        else:
            df = pd.merge(
                filt_samples,
                metadata.loc[:,['label',]],
                how='inner',
                left_index=True,
                right_index=True
            )
        return df
                                     
    def k_fold_split_train_test(self, transient_group, srg):
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

        # Load train and test data (holdout of 10%)
        meta_df = self.retrieve_transient_metadata(transient_group)
        for groups in self.kf.split(meta_df.index, meta_df['label']):
            train_df, test_df = self.split(meta_df, split_indices=groups)
            train_df, val_df = self.split(train_df, split_frac=0.1)
            train_srg = srg.filter(train_df.index)
            val_srg = srg.filter(val_df.index)
            test_srg = srg.filter(test_df.index)
            k_fold_datasets.append((
                (train_df, train_srg),
                (val_df, val_srg),
                (test_df, test_srg)
            ))
            
        return k_fold_datasets
    
    def split(self, all_data, split_frac=0.1, split_indices=None):
        
        if split_indices is not None:
            idx1, idx2 = split_indices
        else:
            idx1, idx2 = train_test_split(
                np.arange(len(all_data)),
                stratify=all_data['label'],
                test_size=split_frac,
                random_state=self.config.random_seed
            )
            
        return all_data.iloc[idx1,:], all_data.iloc[idx2,:]
        
    def split_train_test(self, transient_group, srg):
        """Reads data and splits it into training and testing sets.

        Parameters
        ----------
        transient_group: TransientGroup
        all information about transient set
        
        Returns
        -------
        tuple
            The train data and the test data.
        """
        meta_df = self.retrieve_transient_metadata(transient_group)
        train_df, test_df = self.split(meta_df, split_frac=0.1)
        train_df, val_df = self.split(train_df, split_frac=0.1)
        train_srg = srg.filter(train_df.index)
        val_srg = srg.filter(val_df.index)
        test_srg = srg.filter(test_df.index)
        
        return (
            (train_df, train_srg),
            (val_df, val_srg),
            (test_df, test_srg)
        )
