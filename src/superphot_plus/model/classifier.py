import pickle

import numpy as np
import pandas as pd
import lightgbm
from torch.utils.data import DataLoader

from ..constants import EPOCHS
from ..config import SuperphotConfig
from .metrics import ModelMetrics

class SuperphotClassifier:
    """Base classifier model.

    Parameters
    ----------
    config : SuperphotConfig
        The MLP architecture configuration.
    """

    def __init__(self, config: SuperphotConfig):
        super().__init__()

        # Initialize MLP architecture        
        # Model state dictionary
        self.best_model = None
        self.best_val_loss = np.inf
        self.target_label = config.target_label
        self.prob_threshhold = config.prob_threshhold
        self.probs_fn = config.probs_fn
        self.normalization_means = None
        self.normalization_stddevs = None
    
    def normalize(self, features):
        """Normalize features according to self.normalization_means
        and self.normalization_stddevs if not None. If they are None,
        perform StandardScaler normalization.
        """
        if (self.normalization_means is None):
            self.normalization_means = features.mean(axis=0)
        if self.normalization_stddevs is None:
            self.normalization_stddevs = features.std(axis=0)
        return (features - self.normalization_means) / self.normalization_stddevs
    
    def train_and_validate(
        self,
        train_data,
        val_data,
        num_epochs=EPOCHS,
        rng_seed=None,
        **kwargs,
    ):
        """
        Runs LightGBM training and validation.
        
        Parameters
        ----------
        train_data : TrainData
            The training dataset.
        rng_seed : int, optional
            Random state that is seeded. if none, use machine entropy.

        Returns
        -------
        tuple
            A tuple containing arrays of metrics for each epoch
            (training accuracies and losses, validation accuracies and losses).
        """
        return ModelMetrics().get_values()
        
    
    def evaluate(self, test_features, normalized=False):
        """Runs model over a group of test samples.

        Parameters
        ----------
        test_data : TestData
            The data to evaluate the model. Consists of test features,
            test classes, test names and a list of grouped indices, respectively.
        probs_csv_path : str, optional
            Where to store the probability results.

        Returns
        -------
        tuple
            A tuple containing the labels, names, predicted labels
            and maximum probabilities.
            
        TODO: GIVE COLUMN NAMES FOR PROBS
        """
        if not normalized:
            test_features = self.normalize(test_features)
            
        probabilities = pd.DataFrame(
            self.best_model.predict_proba(test_features),
            index=test_features.index
        )
        probs_avg = probabilities.groupby(test_features.index).mean()
        
        return probs_avg
        
    def set_best_val_loss(self, best_val_loss):
        """Sets the best validation loss from training."""
        self.best_val_loss = best_val_loss
            
    def save(self, config_prefix):
        """Save the classifier as file.

        Parameters
        ----------
        models_dir : str
            Directory to write to
        """
        with open(f"{config_prefix}.pt", 'wb') as f:
            pickle.dump(self, f)
        
    @classmethod
    def create(cls, config):
        """Creates a Model instance.

        Parameters
        ----------
        config : ModelConfig
            Includes (in order): input_size, output_size, n_neurons, n_hidden.
            Also includes normalization means and standard deviations.

        Returns
        ----------
        torch.nn.Module
            The MLP object.
        """
        return cls(config)
            
    @classmethod
    def load(cls, filename):
        """Load a classifier that was saved to disk

        Parameters
        ----------
        path : str
            Path where the classifier was saved

        Returns
        -------
        `~Classifier`
            Loaded classifier
        """
        with open(filename, 'rb') as f:
            model = pickle.load(f)

        return model
