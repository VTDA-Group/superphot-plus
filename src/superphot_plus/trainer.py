import os

import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split

from superphot_plus.format_data_ztf import (
    normalize_features,
    tally_each_class,
    retrieve_posterior_set
)
from superphot_plus.model.mlp import SuperphotMLP
from superphot_plus.model.lightgbm import SuperphotLightGBM

from superphot_plus.config import SuperphotConfig
from superphot_plus.model.data import TestData, TrainData, PosteriorSamplesGroup
from superphot_plus.plotting.classifier_results import plot_model_metrics
from superphot_plus.plotting.confusion_matrices import plot_matrices
from superphot_plus.supernova_class import SupernovaClass as SnClass
from superphot_plus.trainer_base import TrainerBase
from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.utils import (
    create_dataset,
    extract_wrong_classifications,
    log_metrics_to_tensorboard,
    write_metrics_to_file,
    epoch_time,
    save_test_probabilities,
)


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
    include_redshift : bool
        If True, includes redshift data for training.
    probs_file : str
        The file where test probabilities are written. Defaults to PROBS_FILE.
    """

    def __init__(
        self,
        config_name,
        fits_dir,
        sampler="dynesty",
        model_type='LightGBM',
        include_redshift=True,
        probs_file=None,
        n_folds=10,
        target_label=None,
    ):
        super().__init__(
            config_name, fits_dir,
            sampler, model_type,
            include_redshift, probs_file, n_folds,
            target_label
        )

    def setup_model(self, load_checkpoint=False):
        """Reads model configuration from disk and loads the
        saved checkpoint if load_checkpoint flag was enabled.

        Parameters
        ----------
        load_checkpoint : bool
            If true, load pretrained model checkpoint.
        """
        config = SuperphotConfig.from_file(self.config_name)
        path = os.path.join(config.models_dir, self.config_name.split('/')[-1].split('.')[0])
                
        if self.probs_file is not None:
            config.probs_fn = self.probs_file
            
        for i in range(self.n_folds):
            if load_checkpoint:
                model_file, config_file = f"{path}_{i}.pt", f"{path}_{i}.yaml"
                model_i, config_i = self._load_model_instance(model_file, config_file)
                self.models.append(model_i)
                self.configs.append(config_i)
            else:
                self.models.append(None)
                self.configs.append(copy.deepcopy(config))
                self.configs[-1].probs_fn = config.probs_fn % i

        self.load_checkpoint = load_checkpoint

    def _load_model_instance(self, model_file, config_file):
        if self.model_type == 'LightGBM':
            return SuperphotLightGBM.load(model_file, config_file)
        elif self.model_type == 'MLP':
            return SuperphotMLP.load(model_file, config_file)
        else:
            raise ValueError
    
    def _create_model_instance(self, config):
        if self.model_type == 'LightGBM':
            return SuperphotLightGBM(config, target_label=self.target_label)
        elif self.model_type == 'MLP':
            return SuperphotMLP.create(config)
        else:
            raise ValueError
    
    def run(self, input_csvs=None, extract_wc=False, n_folds=1, load_checkpoint=False):
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

        if self.n_folds <= 1:
            k_folded_data = [self.split_train_test(input_csvs),]
            
        k_folded_data = self.k_fold_split_train_test(self.kf, input_csvs)
        
        for i in range(self.n_folds):
            print(f"Running fold {i}")
            train_data, test_data = k_folded_data[i]
            self.train(i, train_data)
            # Evaluate model on test dataset
            self.evaluate(i, test_data, extract_wc)
            
        # concatenate probs csvs
        concat_path = self.probs_file.replace("%d", "%s") % "concat"
        concat_df = pd.read_csv(self.probs_file % 0)

        concat_df['Fold'] = 0
        for i in range(1, 10):
            new_df = pd.read_csv(self.probs_file % i)
            new_df['Fold'] = i

            concat_df = pd.concat(
                [concat_df,
                new_df],
                ignore_index=True
            )
        concat_df.to_csv(concat_path, index=False)

    def classify_single_light_curve(self, obj_name, fits_dir, sampler="dynesty"):
        """Given an object name, return classification probabilities
        based on the model fit and data.

        Parameters
        ----------
        obj_name : str
            Name of the supernova.
        fits_dir : str
            Where model fit information is stored.
        sampler : str
            The MCMC sampler to use. Defaults to "dynesty".

        Returns
        ----------
        np.ndarray
            The average probability for each SN type across all equally-weighted sets of fit parameters.
        """
        post_features = get_posterior_samples(
            obj_name,
            fits_dir,
            sampler
        )[0]
        
        if np.median(post_features[:,-1]) > self.chisq_cutoff:
            return -1 * np.ones(len(self.allowed_types))

        # normalize the log distributions
        post_features = np.delete(post_features, self.skipped_params, 1)
        probs_avg = np.zeros(len(self.allowed_types))
        
        for model in self.models: # ensemble classifier
            probs = model.classify_from_fit_params(post_features)
            probs_avg += np.mean(probs, axis=0)
            
        return probs_avg / self.n_folds
    
    def return_new_classifications(self, test_csv, fit_dir, save_file, output_dir=None, include_labels=False, sampler='dynesty'):
        """Return new classifications based on model and save probabilities
        to a CSV file.

        Parameters
        ----------
        test_csv : str
            Path to the CSV file containing the test data.
        fit_dir : str
            Path to the directory containing the fit data.
        save_file : str
            File to store the new classification outputs.
        output_dir : str
            Path to the directory to store the classification outputs.
        include_labels : bool, optional
            If True, labels from the test data are included in the
            probability saving process. Defaults to False.
        """
        filepath = save_file if output_dir is None else os.path.join(output_dir, save_file)

        df = pd.read_csv(test_csv)
        names = df.NAME.to_numpy()
        try:
            redshifts = df.Z.to_numpy()
        except:
            redshifts = -1 * np.ones(len(names))
        
        if not self.include_redshift:
            redshifts = None
            
        if include_labels:
            labels = df.CLASS.to_numpy()
        else:
            labels = None
            
        posts = retrieve_posterior_set(
            names, fit_dir, sampler=sampler,
            redshifts=redshifts, labels=labels,
            chisq_cutoff=self.chisq_cutoff,
        )
        test_data = PosteriorSamplesGroup(
            posts,
            ignore_param_idxs=self.skipped_params,
            use_redshift_info=self.include_redshift,
            random_seed=self.random_seed
        )
        
        combined_probs = np.zeros((len(posts), len(self.allowed_types)))
        for k_fold in range(self.n_folds):
            
            probs = self.models[k_fold].classify_from_fit_params(
                test_data.features
            )
            probs = probs.reshape((
                len(posts),
                test_data.num_draws,
                len(self.allowed_types)
            ))
            combined_probs += np.mean(probs, axis=1)
            
        save_test_probabilities(
            test_data.names,
            combined_probs / self.n_folds,
            filepath,
            true_labels=test_data.labels,
            target_label=self.target_label
        )
        
            
    def train(self, i: int, train_data: PosteriorSamplesGroup):
        """Trains the model with a specific set of hyperparameters.

        Parameters
        ----------
        i : the k-fold index
        train_data : PosteriorSamplesGroup
            Contains the ZTF object names, classes and redshifts for training.
        """
        run_id = f"final_{i}"
        #tally_each_class(train_data.labels)  # original tallies

        # Split data into training and validation sets
        train_index, val_index = train_test_split(
            np.arange(0, len(train_data.labels)),
            stratify=train_data.labels,
            test_size=0.1
        )

        train_features, train_classes, val_features, val_classes = self.generate_train_data(
            train_data=train_data,
            goal_per_class=self.configs[0].goal_per_class,
            train_index=train_index,
            val_index=val_index,
        )
        
        train_features, mean, std = normalize_features(train_features)
        val_features, mean, std = normalize_features(val_features, mean, std)

        train_dataset = create_dataset(train_features, train_classes)
        val_dataset = create_dataset(val_features, val_classes)

        if not self.load_checkpoint:
            self.configs[i].set_non_tunable_params(
                input_dim=train_features.shape[1],
                output_dim=len(self.allowed_types),
                norm_means=mean.tolist(),
                norm_stddevs=std.tolist(),
            )
            self.models[i] = self._create_model_instance(self.configs[i])

        # Train and validate multi-layer perceptron
        metrics = self.models[i].train_and_validate(
            train_data=TrainData(train_dataset, val_dataset),
            rng_seed=self.random_seed,
            num_epochs=self.configs[i].num_epochs,
        )

        # Save model checkpoint
        prefix = os.path.join(self.configs[i].models_dir, self.config_name.split('/')[-1].split('.')[0])
        self.models[i].save(prefix, suffix=i)

        # Plot training and validation metrics
        plot_model_metrics(
            metrics=metrics,
            plot_name=run_id,
            metrics_dir=self.configs[i].metrics_dir,
        )

        # Log average metrics per epoch to plot on Tensorboard.
        #log_metrics_to_tensorboard(metrics=[metrics], config=self.configs[i], trial_id=run_id)

    def evaluate(self, k_fold, test_data: PosteriorSamplesGroup, extract_wc=False):
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
        if self.models[k_fold] is None:
            raise ValueError("Cannot evaluate uninitialized model.")

        test_features, test_classes, test_names = self.generate_test_data(
            test_data=test_data
        )
        
        mean = self.configs[k_fold].normalization_means
        std = self.configs[k_fold].normalization_stddevs
        
        test_features, _, _ = normalize_features(test_features, mean, std)

        results = self.models[k_fold].evaluate(
            test_data=TestData(test_features, test_classes, test_names),
        )

        true_classes, _, pred_classes, pred_probs = zip(results)

        true_classes = np.hstack(true_classes)
        pred_classes = np.hstack(pred_classes)
        pred_probs_above_07 = np.hstack(pred_probs) > 0.7

        true_classes = SnClass.get_labels_from_classes(true_classes)
        pred_classes = SnClass.get_labels_from_classes(pred_classes)

        # Log evaluation metrics
        write_metrics_to_file(
            config=self.configs[k_fold],
            true_classes=true_classes,
            pred_classes=pred_classes,
            prob_above_07=pred_probs_above_07,
        )
        plot_matrices(
            config=self.configs[k_fold],
            true_classes=true_classes,
            pred_classes=pred_classes,
            prob_above_07=pred_probs_above_07,
        )
        if extract_wc:
            extract_wrong_classifications(
                true_classes=true_classes,
                pred_classes=pred_classes,
                ztf_test_names=test_data.names,
            )
