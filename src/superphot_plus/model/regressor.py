"""This module implements the Multi-Layer Perceptron (MLP) model for classification."""
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from superphot_plus.constants import EPOCHS
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.metrics import RegressorMetrics
from superphot_plus.model.mlp import SuperphotMLP
from superphot_plus.utils import epoch_time


class SuperphotRegressor(SuperphotMLP):
    """Implements regression MLP to estimate supernova physical parameters."""

    def __init__(self, config: ModelConfig):
        """Implements regression MLP to estimate supernova physical parameters.

        Parameters
        ----------
        config : ModelConfig
            The neural network specification.
        """
        super().__init__(config, nn.MSELoss())

    def train_and_validate(
        self,
        train_data,
        num_epochs=EPOCHS,
        rng_seed=None,
    ):
        """Runs training for the regressor.

        Parameters
        ----------
        train_data : TrainData
            The training and validation datasets.
        num_epochs : int, optional
            The number of epochs. Defaults to EPOCHS.
        rng_seed : int, optional
            Random state that is seeded. if none, use machine entropy.

        Returns
        -------
        tuple
            Training and validation losses, for each epoch.
        """
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed(rng_seed)
            torch.backends.cudnn.deterministic = True

        train_dataset, valid_dataset = train_data

        train_iterator = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
        )
        valid_iterator = DataLoader(
            dataset=valid_dataset,
            batch_size=self.config.batch_size,
        )

        metrics = RegressorMetrics()

        best_model = None
        best_val_loss = float("inf")

        for epoch in np.arange(0, num_epochs):
            start_time = time.monotonic()

            train_loss = self.train_epoch(train_iterator)
            val_loss = self.evaluate_epoch(valid_iterator)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.state_dict()

            end_time = time.monotonic()

            # Store metrics for the current epoch
            metrics.append(
                train_loss=train_loss,
                val_loss=val_loss,
                epoch_time=epoch_time(start_time, end_time),
            )

            if epoch % 5 == 0:
                metrics.print_last()

        # Save best model state
        self.best_model = best_model
        self.load_state_dict(best_model)

        # Store best validation loss
        self.config.set_best_val_loss(best_val_loss)

        return metrics.get_values()

    def train_epoch(self, iterator):
        """Does one epoch of training for a given model.

        Parameters
        ----------
        iterator : torch.utils.DataLoader
            The data iterator.

        Returns
        -------
        tuple
            A tuple containing the epoch training loss.
        """
        epoch_loss = 0

        self.train()

        for x, y in iterator:
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            self.optimizer.zero_grad()

            y_pred, _ = self(x)

            loss = self.criterion(y_pred, y.float())

            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def evaluate_epoch(self, iterator):
        """Does one epoch of validation for a given model.

        Parameters
        ----------
        iterator : torch.utils.DataLoader
            The data iterator.

        Returns
        -------
        tuple
            A tuple containing the epoch validation loss.
        """
        epoch_loss = 0

        self.eval()

        with torch.no_grad():
            for x, y in iterator:
                x = x.to(self.config.device)
                y = y.to(self.config.device)

                y_pred, _ = self(x)

                loss = self.criterion(y_pred, y.float())

                epoch_loss += loss.item()

        return epoch_loss / len(iterator)

    def get_predictions(self, iterator):
        """Given a trained model, returns the physical
        parameter predictions across all the test labels.

        Parameters
        ----------
        iterator : torch.utils.DataLoader
            The data iterator.

        Returns
        -------
        tuple
            A tuple containing the model predictions.
        """
        self.eval()

        predictions = []

        with torch.no_grad():
            for x, _ in iterator:
                x = x.to(self.config.device)

                y_pred, _ = self(x)

                predictions.append(y_pred.float())

        predictions = torch.cat(predictions, dim=0)

        return np.array(predictions)

    @classmethod
    def create(cls, config):
        """Creates an MLP instance, optimizer and respective criterion.

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
        assert config.output_dim == 1
        model = cls(config)
        model.criterion = model.criterion.to(config.device)
        model = model.to(config.device)
        return model

    @classmethod
    def load(cls, filename, config_filename):
        """Load a trained MLP for subsequent regression of new objects.

        Parameters
        ----------
        filename : str
            The path to the pre-trained model.
        config_filename : str
            The file that includes the model training configuration.

        Returns
        ----------
        tuple
            The pre-trained regressor object and the respective model config.
        """
        config = ModelConfig.from_file(config_filename)
        model = SuperphotRegressor.create(config)  # set up empty multi-layer perceptron
        model.load_state_dict(torch.load(filename))  # load trained state dict to the MLP
        return model, config
