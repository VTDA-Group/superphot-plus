from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocess as mp
from snapi import TransientGroup, SamplerResultGroup

from .model.mlp import SuperphotMLP
from .model.lightgbm import SuperphotLightGBM
from .plotting.confusion_matrices import plot_matrices
from .trainer_base import TrainerBase

class SuperphotTrainer(TrainerBase):
    """
    Trains and evaluates models using K-Fold cross validation.

    The model may be trained from scratch using a specified configuration
    or be loaded from a previous checkpoint stored on disk. In both scenarios
    the model is evaluated on a test holdout set and metrics are generated.

    Parameters
    ----------
    config_name : str
        The name of the pre-trained model configuration to load. This file should
        be located under the specified models directory. Defaults to None.
    sampler : str
        The type of sampler used for the lightcurve fits. Defaults to "dynesty".
    """
    def setup_model(self):
        """Reads model configuration from disk and loads the
        saved checkpoint if load_checkpoint flag was enabled.

        Parameters
        ----------
        load_checkpoint : bool
            If true, load pretrained model checkpoint.
        """
            
        for i in range(self.config.n_folds):
            if self.config.load_checkpoint:
                model_i = self._load_model_instance(f"{self.config.model_prefix}_{i}.pt")
                self.models.append(model_i)
            else:
                self.models.append(None)

    def _load_model_instance(self, model_file):
        if self.config.model_type == 'LightGBM':
            return SuperphotLightGBM.load(model_file)
        elif self.config.model_type == 'MLP':
            return SuperphotMLP.load(model_file)
        else:
            raise ValueError
    
    def _create_model_instance(self):
        if self.config.model_type == 'LightGBM':
            return SuperphotLightGBM(self.config)
        elif self.config.model_type == 'MLP':
            return SuperphotMLP.create(self.config)
        else:
            raise ValueError
    
    def run_single_fold(self, data):
        """Run single fold training + evaluation."""
        i, (train_data, val_data, test_data) = data
        self.train(i, train_data, val_data)
        probs_df = self.evaluate(i, test_data)
        return probs_df
        
    def run(
        self,
        transient_data: Optional[TransientGroup] = None,
        sampler_results: Optional[SamplerResultGroup] = None,
    ):
        """Runs the machine learning workflow.

        Trains the model on the whole training set and evaluates it on a
        test holdout set. Also allows K-fold cross-validation.
        Metrics are plotted and logged to files.

        Parameters
        ----------
        input_csvs : list of str
            The list of training CSV files. Defaults to INPUT_CSVS.
        extract_wc : bool
            If true, assumes all sample fit plots are saved in
            FIT_PLOTS_FOLDER. Copies plots of wrongly classified samples to
            separate folder for manual followup. Defaults to False.
        load_checkpoint : bool
            If true, load pretrained model checkpoint.
        """
        # Loads model and config
        self.setup_model()
        
        if sampler_results is None:
            sampler_results = SamplerResultGroup.load(self.config.sampler_results_fn)
        if transient_data is None:
            transient_data = TransientGroup.load(self.config.transient_data_fn)
        
        if self.config.n_folds <= 1:
            k_folded_data = [self.split_train_test(transient_data, sampler_results),]
        else:
            k_folded_data = self.k_fold_split_train_test(transient_data, sampler_results)

        if self.config.n_parallel > 1:
            ctx = mp.get_context('spawn')
            pool = ctx.Pool(self.config.n_parallel)
            probs_df = pool.map(self.run_single_fold, zip(np.arange(self.config.n_folds), k_folded_data))
        else:
            probs_df = []
            for fold in range(self.config.n_folds):
                probs_df.append(self.run_single_fold(fold, k_folded_data[fold]))
                
        concat_df = pd.concat(probs_df)
        concat_df.to_csv(self.config.probs_fn)
        
        if self.config.plot: # Plot joint confusion matrix
            plot_matrices(self.config, concat_df)
            
    def return_new_classifications(
        self,
        transient_group: TransientGroup,
        sr_group: SamplerResultGroup,
        save_fn: str
    ):
        """Return classifications for new set of events.
        """
        transient_group = TransientGroup.load(self.config.transient_data_fn)
        meta_df = self.retrieve_transient_metadata(transient_group)
        
        concat_df = None
        
        for i in range(self.config.n_folds):
            probs_df = self.evaluate(i, (meta_df, sr_group))
            
            # save to probability csv
            if concat_df is None:
                concat_df = probs_df
            else:
                concat_df = pd.concat([concat_df, probs_df], axis=0)
            
        concat_probs = concat_df.drop('pred_class', axis=1)
        probs_avg = concat_probs.groupby(concat_probs.index).mean()
        probs_avg['true_class'] = concat_probs.groupby(concat_probs.index)['true_class'].first()
        
        if self.config.target_label is None:
            probs_avg.columns = np.sort(self.config.allowed_types)
            probs_avg['pred_class'] = probs_avg.idxmax(axis=1)
        else:
            probs_avg.columns = np.sort([self.config.target_label, "other"])
            pred_target = probs_avg[self.config.target_label] > self.config.prob_threshhold
            probs_avg['pred_class'] = np.where(pred_target, self.config.target_label, "other")
        
        probs_avg['fold'] = -1 # all combined
        concat_df = pd.concat([concat_df, probs_avg], ignore_index=False)
        concat_df.to_csv(save_fn)
        
            
    def train(self, i: int, train_data, val_data):
        """Trains the model with a specific set of hyperparameters.

        Parameters
        ----------
        i : the k-fold index
        train_data : PosteriorSamplesGroup
            Contains the ZTF object names, classes and redshifts for training.
        """
        train_df = self.retrieve_sampler_results(train_data[1], train_data[0], balance_classes=True)
        val_df = self.retrieve_sampler_results(val_data[1], val_data[0], balance_classes=True)
                
        if self.config.input_features is None:
            self.config.input_features = train_df.columns[~train_df.columns.isin(['label', 'score', 'sampler'])]

        # extract features
        train_features = train_df.loc[:, self.config.input_features]
        val_features = val_df.loc[:, self.config.input_features]
        
        if not self.config.load_checkpoint:
            self.models[i] = self._create_model_instance()

        # Train and validate multi-layer perceptron
        metrics = self.models[i].train_and_validate(
            train_data=(train_features, train_df['label']),
            val_data=(val_features, val_df['label']),
            rng_seed=self.config.random_seed,
            num_epochs=self.config.num_epochs,
        )

        # Save model checkpoint
        self.models[i].save(self.config.model_prefix + f"_{i}")
                
        if self.config.plot:
            # Plot training and validation metrics
            fig, ax = plt.subplots(1, 2, figsize=(12,6))
            metrics.plot(ax=ax)
            fig.savefig(self.config.metrics_prefix + f"_{i}.pdf")
            plt.close()

        
    def evaluate(self, k_fold, test_data):
        """Evaluates a pretrained model on the test holdout set.

        Parameters
        ----------
        test_data : PosteriorSamplesGroup
            Contains the ZTF object names, classes and redshifts for testing.
        extract_wc : bool
            If true, assumes all sample fit plots are saved in
            FIT_PLOTS_FOLDER. Copies plots of wrongly classified samples to
            separate folder for manual followup. Defaults to False.
        Returns
        -------
        tuple
            A tuple containing the test ground truths, the respective
            predicted classes and the predicted classes for which
            classification confidence exceeded 70%.
        """
        model = self.models[k_fold]
        if model is None:
            raise ValueError("Cannot evaluate uninitialized model.")
            
        test_df = self.retrieve_sampler_results(test_data[1], test_data[0])

        if self.config.input_features is None:
            self.config.input_features = test_df.columns[~test_df.columns.isin(['label', 'score', 'sampler'])]

        probs_avg = model.evaluate(test_df[self.config.input_features])
        
        if self.config.target_label is None:
            probs_avg.columns = np.sort(self.config.allowed_types)
            probs_avg['pred_class'] = probs_avg.idxmax(axis=1)
        else:
            probs_avg.columns = np.sort([self.config.target_label, "other"])
            pred_target = probs_avg[self.config.target_label] > self.config.prob_threshhold
            probs_avg['pred_class'] = np.where(pred_target, self.config.target_label, "other")

        probs_avg['true_class'] = test_df['label'].groupby(test_df.index).first()
        probs_avg['fold'] = k_fold
        """
        if self.config.logging:
            # Log evaluation metrics
            write_metrics_to_file(
                self.config, probs_avg
            )
        if self.config.plot:
            plot_matrices(
                self.config, probs_avg
            )
        """
        return probs_avg
