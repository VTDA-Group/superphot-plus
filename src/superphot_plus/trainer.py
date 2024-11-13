import os
import copy


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from snapi.analysis import SamplerResult

from superphot_plus.model.mlp import SuperphotMLP
from superphot_plus.model.lightgbm import SuperphotLightGBM
from superphot_plus.config import SuperphotConfig
from superphot_plus.model.data import TestData, TrainData, PosteriorSamplesGroup
from superphot_plus.plotting.classifier_results import plot_model_metrics
from superphot_plus.plotting.confusion_matrices import plot_matrices
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.trainer_base import TrainerBase
from superphot_plus.utils import write_metrics_to_file


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
    def setup_model(self, load_checkpoint=False):
        """Reads model configuration from disk and loads the
        saved checkpoint if load_checkpoint flag was enabled.

        Parameters
        ----------
        load_checkpoint : bool
            If true, load pretrained model checkpoint.
        """
        model_path = os.path.join(self.config.models_dir, self.config.prefix)
            
        for i in range(self.config.n_folds):
            if load_checkpoint:
                model_i = self._load_model_instance(f"{path}_{i}.pt")
                self.models.append(model_i)
            else:
                self.models.append(None)

        self.load_checkpoint = load_checkpoint

    def _load_model_instance(self, model_file):
        if self.model_type == 'LightGBM':
            return SuperphotLightGBM.load(model_file)
        elif self.model_type == 'MLP':
            return SuperphotMLP.load(model_file)
        else:
            raise ValueError
    
    def _create_model_instance(self, config=None):
        if config is None:
            config = self.config
        if config.model_type == 'LightGBM':
            return SuperphotLightGBM(config)
        elif config.model_type == 'MLP':
            return SuperphotMLP.create(config)
        else:
            raise ValueError
    
    def run(
        self,
        transient_data: Optional[TransientGroup] = None,
        sampler_results: Optional[SamplerResultGroup] = None,
        load_checkpoint=False
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
        self.setup_model(load_checkpoint)
        
        if sampler_results is None:
            sampler_results = SamplerResultGroup.load(self.config.sampler_results_fn)
        if transient_data is None:
            transient_data = TransientGroup.load(self.config.transient_data_fn)
        
        if self.config.n_folds <= 1:
            k_folded_data = [self.split_train_test(transient_group, srg),]
        else:
            k_folded_data = self.k_fold_split_train_test(transient_group, srg)
            
        concat_df = None
        
        for i in range(self.config.n_folds):
            print(f"Running fold {i}")
            train_data, val_data, test_data = k_folded_data[i]
            self.train(i, train_data, val_data)
            probs_df = self.evaluate(i, test_data)
            
            # save to probability csv
            if concat_df is None:
                concat_df = probs_df
            else:
                concat_df = pd.concat([concat_df, probs_df], axis=0)
            
        concat_df.to_csv(self.config.probs_fn)
    
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
        
        for i in range(self.n_folds):
            probs_df = self.evaluate(i, meta_df, sr_group)
            
            # save to probability csv
            if concat_df is None:
                concat_df = probs_df
            else:
                concat_df = pd.concat([concat_df, probs_df], axis=0)
            
        concat_probs = concat_df.drop('pred_class', axis=1)
        probs_avg = concat_probs.groupby(concat_probs.index).mean()
        probs_avg['true_class'] = input_df['label']
        
        if self.config.target_label is None:
            probs_avg['pred_class'] = SnClass.get_labels_from_classes(probs_avg.idxmax(axis=1))
        else:
            pred_target = probs_avg.iloc[:,1] > self.prob_threshhold
            probs_avg['pred_class'] = 1 - pred_target.astype(int)
            #TODO: check this
        
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
        train_df = self.retrieve_sampler_results(self, train_data[1], train_data[0], balance_classes=True)
        val_df = self.retrieve_sampler_results(self, val_data[1], val_data[0], balance_classes=True)
        
        # extract features
        train_features = train_df[self.config.input_features]
        val_features = val_df[self.config.input_features]

        if not self.load_checkpoint:
            self.models[i] = self._create_model_instance(self.config)

        # Train and validate multi-layer perceptron
        metrics = self.models[i].train_and_validate(
            train_data=(train_features, train_df['label']),
            val_data=(val_features, val_df['label']),
            rng_seed=self.config.random_seed,
            num_epochs=self.config.num_epochs,
        )

        # Save model checkpoint
        prefix = os.path.join(self.config.models_dir, self.config.prefix)
        self.models[i].save(prefix, suffix=i)
        
        run_id = f"final_{i}"
        if self.config.plot:
            # Plot training and validation metrics
            plot_model_metrics(
                metrics=metrics,
                plot_name=run_id,
                metrics_dir=self.config.metrics_dir,
            )

        if self.config.logging:
            log_metrics_to_tensorboard(metrics=[metrics], config=self.config, trial_id=run_id)

        
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

        test_df = self.retrieve_sampler_results(self, test_data[1], test_data[0])
        probs_avg = model.evaluate(test_data=test_df[self.config.input_features])
        
        if self.config.target_label is None:
            probs_avg['pred_class'] = SnClass.get_labels_from_classes(probs_avg.idxmax(axis=1))
        else:
            pred_target = probs_avg.iloc[:,1] > self.prob_threshhold
            probs_avg['pred_class'] = 1 - pred_target.astype(int)
            #TODO: check this
            
        probs_avg['true_class'] = SnClass.get_labels_from_classes(test_df['label'])
        probs_avg['fold'] = i
        
        if self.config.logging:
            # Log evaluation metrics
            write_metrics_to_file(
                config=self.config,
                probs_avg
            )
        if self.config.plot:
            plot_matrices(
                config=self.config,
                probs_avg
            )        
        return probs_avg
