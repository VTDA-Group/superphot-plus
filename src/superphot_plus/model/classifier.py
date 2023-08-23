"""This module implements the Multi-Layer Perceptron (MLP) model for
classification."""

import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from superphot_plus.constants import (
    EPOCHS,
    HIDDEN_DROPOUT_FRAC,
    INPUT_DROPOUT_FRAC,
    SEED,
)
from superphot_plus.file_paths import METRICS_DIR, MODELS_DIR, PROBS_FILE
from superphot_plus.format_data_ztf import normalize_features
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.data import TrainData, TestData
from superphot_plus.model.metrics import ModelMetrics
from superphot_plus.plotting.classifier_results import plot_model_metrics
from superphot_plus.utils import calculate_accuracy, create_dataset, epoch_time, save_test_probabilities

from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset


class SuperphotClassifier(nn.Module):
    """The Multi-Layer Perceptron.

    Parameters
    ----------
    config : ModelConfig
        The MLP architecture configuration.
    train_data : TrainData
        The training and validation datasets.
    test_data : TestData, optional
        The testing data.
    """

    def __init__(
        self,
        config: ModelConfig,
        train_data: TrainData,
        test_data: TestData = None
    ):
        super().__init__()

        # Initialize MLP architecture
        self.config = config

        (
            input_dim,
            output_dim,
            neurons_per_layer,
            num_hidden_layers,
            _,
            learning_rate
        ) = config

        n_neurons = neurons_per_layer
        self.input_fc = nn.Linear(input_dim, n_neurons)

        assert num_hidden_layers >= 1

        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.dropouts.append(nn.Dropout(INPUT_DROPOUT_FRAC))

        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(n_neurons, n_neurons))
        for _ in range(num_hidden_layers):
            self.dropouts.append(nn.Dropout(HIDDEN_DROPOUT_FRAC))

        self.output_fc = nn.Linear(n_neurons, output_dim)

        # Optimizer and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Set training and testing data
        self.train_data = train_data
        self.test_data = test_data

    def forward(self, x):
        """Forward pass of the Multi-Layer Perceptron model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        tuple
            A tuple containing the predicted output tensor and the
            hidden tensor.
        """
        batch_size = x.shape[0]

        x = x.view(batch_size, -1)

        h_1 = self.dropouts[0](x)
        h_1 = F.relu(self.input_fc(h_1))

        h_hidden = h_1
        for i, layer in enumerate(self.hidden_layers):
            h_hidden = self.dropouts[i + 1](h_hidden)
            h_hidden = F.relu(layer(h_hidden))

        h_hidden = self.dropouts[-1](h_hidden)
        y_pred = self.output_fc(h_hidden)

        return y_pred, h_hidden

    def run(
        self,
        run_id="run_0",
        num_epochs=EPOCHS,
        plot_metrics=False,
        metrics_dir=METRICS_DIR,
        models_dir=MODELS_DIR,
        probs_csv_path=PROBS_FILE,
    ):
        """
        Run the MLP initialization and training.

        Closely follows the demo
        https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

        Parameters
        ----------
        run_id: int, optional
            A call identifier (useful for parallel calls on cross-validation).
        num_epochs : int, optional
            The number of epochs. Defaults to EPOCHS.
        plot_metrics : bool, optional
            Whether to plot metrics. Defaults to False.
        metrics_dir : str, optional
            Where to store metrics.
        models_dir : str, optional
            Where to store models.
        probs_csv_path :
            Where to store the probability results.

        Returns
        -------
        tuple
            A tuple containing the labels, names, predicted labels, maximum
            probabilities, and best validation loss.
        """
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.backends.cudnn.deterministic = True

        (
            train_dataset,
            valid_dataset,
        ) = self.train_data

        config_path = os.path.join(models_dir, f"superphot-config-{run_id}.json")
        self.config.save(config_path)

        model_path = os.path.join(models_dir, f"superphot-model-{run_id}.pt")

        train_iterator = DataLoader(train_dataset, shuffle=True, batch_size=self.config.batch_size)
        valid_iterator = DataLoader(valid_dataset, batch_size=self.config.batch_size)

        best_valid_loss = float("inf")
        valid_acc = float("inf")

        metrics = ModelMetrics()

        for epoch in np.arange(0, num_epochs):
            start_time = time.monotonic()

            train_loss, train_acc = self.train_epoch(train_iterator)
            val_loss, val_acc = self.evaluate_epoch(valid_iterator)

            if val_loss < best_valid_loss:
                best_valid_loss = val_loss
                valid_acc = val_acc
                torch.save(self.state_dict(), model_path)

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            # Store metrics for the current epoch
            metrics.append(train_loss, train_acc, val_loss, val_acc, epoch_mins, epoch_secs)

            if epoch % 5 == 0:
                metrics.print_last()

        # Load model with best validation score
        self.load_state_dict(torch.load(model_path))

        # Plot training and validation metrics
        if plot_metrics:
            plot_model_metrics(
                metrics=metrics.get_values(),
                num_epochs=num_epochs,
                plot_name=run_id,
                metrics_dir=metrics_dir,
            )

        if self.test_data is None:
            return best_valid_loss, valid_acc

        labels, names, pred_labels, max_probs = self.test(probs_csv_path)

        return (
            np.array(labels).astype(int),
            np.array(names),
            np.array(pred_labels).astype(int),
            np.array(max_probs).astype(float),
            best_valid_loss,
        )

    def train_epoch(self, iterator):
        """Does one epoch of training for a given torch model.

        Parameters
        ----------
        iterator : torch.utils.DataLoader
            The data iterator.

        Returns
        -------
        tuple
            A tuple containing the epoch loss and epoch accuracy.
        """
        epoch_loss = 0
        epoch_acc = 0

        self.train()

        for x, y in iterator:
            x = x.to(self.config.device)
            y = y.to(self.config.device)

            self.optimizer.zero_grad()

            y_pred, _ = self(x)

            loss = self.criterion(y_pred, y)
            acc = calculate_accuracy(y_pred, y)

            loss.backward()

            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def evaluate_epoch(self, iterator):
        """Evaluates the model for the validation set.

        Parameters
        ----------
        iterator : torch.utils.DataLoader
            The data iterator.

        Returns
        -------
        tuple
            A tuple containing the epoch loss and epoch accuracy.
        """
        epoch_loss = 0
        epoch_acc = 0

        self.eval()

        with torch.no_grad():
            for x, y in iterator:
                x = x.to(self.config.device)
                y = y.to(self.config.device)

                y_pred, _ = self(x)
                loss = self.criterion(y_pred, y)

                acc = calculate_accuracy(y_pred, y)

                epoch_loss += loss.item()
                epoch_acc += acc.item()

        return epoch_loss / len(iterator), epoch_acc / len(iterator)

    def test(self, save_path):
        """Runs model over a group of test samples

        Parameters
        ----------
        save_path
            Where to store the probability results.

        Returns
        -------
        tuple
            A tuple containing the labels, names, predicted labels
            and maximum probabilities.
        """
        (
            test_features,
            test_classes,
            test_names,
            test_group_idxs
        ) = self.test_data

        labels, pred_labels, max_probs, names = [], [], [], []

        for group_idx_set in test_group_idxs:
            test_data = create_dataset(
                test_features[group_idx_set],
                test_classes[group_idx_set],
                group_idx_set,
            )

            test_iterator = DataLoader(test_data, batch_size=self.config.batch_size)

            _, labels_indiv, indx_indiv, probs = self.get_predictions(test_iterator)
            probs_avg = np.mean(probs.numpy(), axis=0)

            save_test_probabilities(
                test_names[indx_indiv.numpy().astype(int)[0]],
                probs_avg,
                labels_indiv[0],
                save_file=save_path,
            )

            pred_labels.append(np.argmax(probs_avg))
            max_probs.append(np.amax(probs_avg))
            labels.append(labels_indiv[0])
            names.append(test_names[indx_indiv.numpy().astype(int)[0]])

        return labels, names, pred_labels, max_probs

    def get_predictions(self, iterator):
        """Given a trained model, returns the test images, test labels, and
        prediction probabilities across all the test labels.

        Parameters
        ----------
        iterator : torch.utils.DataLoader
            The data iterator.

        Returns
        -------
        tuple
            A tuple containing the test images, test labels, sample indices,
            and prediction probabilities.
        """
        self.eval()

        images = []
        labels = []
        probs = []
        sample_idxs = []

        with torch.no_grad():
            for x, y, z in iterator:
                x = x.to(self.config.device)

                y_pred, _ = self(x)

                y_prob = F.softmax(y_pred, dim=-1)

                images.append(x.cpu())
                labels.append(y.cpu())
                sample_idxs.append(z.cpu())
                probs.append(y_prob.cpu())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)
        sample_idxs = torch.cat(sample_idxs, dim=0)

        return images, labels, sample_idxs, probs

    def get_predictions_from_fit_params(self, iterator):
        """Given a trained model, returns the test images, test labels, and
        prediction probabilities across all the test labels.

        Parameters
        ----------
        iterator : torch.utils.DataLoader
            The data iterator.

        Returns
        -------
        tuple
            A tuple containing the test images and prediction probabilities.
        """
        self.eval()

        images = []
        probs = []

        with torch.no_grad():
            for x in iterator:
                x = x[0].to(self.config.device)

                y_pred, _ = self(x)

                y_prob = F.softmax(y_pred, dim=-1)

                images.append(x.cpu())
                probs.append(y_prob.cpu())

        images = torch.cat(images, dim=0)
        probs = torch.cat(probs, dim=0)

        return images, probs

    def classify_from_fit_params(self, fit_params):
        """Classify one or multiple light curves solely from the fit parameters
        used in the classifier. Excludes t0 and, for redshift-exclusive
        classifier, A. Includes chi-squared value.

        Parameters
        ----------
        fit_params : np.ndarray
            Set of model fit parameters.

        Returns
        ----------
        np.ndarray
            Probability of each light curve being each SN type.
            Sums to 1 along each row.
        """
        fit_params_2d = np.atleast_2d(fit_params)  # cast to 2D if only 1 light curve
        test_features, _, _ = normalize_features(
            fit_params_2d,
            self.config.normalization_means,
            self.config.normalization_stddevs,
        )
        test_data = TensorDataset(torch.Tensor(test_features))
        test_iterator = DataLoader(test_data, batch_size=self.config.batch_size)
        _, probs = self.get_predictions_from_fit_params(test_iterator)
        return probs.numpy()

    @classmethod
    def create(cls, config, train_data=None, test_data=None):
        """Creates an MLP instance, optimizer and respective criterion.

        Parameters
        ----------
        config : ModelConfig
            Includes (in order): input_size, output_size, n_neurons, n_hidden.
        train_data : TrainData
            Training and validation data.
        test_data : TestData
            Testing data.

        Returns
        ----------
        torch.nn.Module
            The MLP object.
        """
        model = cls(config, train_data, test_data)
        model.criterion = model.criterion.to(config.device)
        model = model.to(config.device)
        return model

    @classmethod
    def load(cls, filename, config_filename):
        """Load a trained MLP for subsequent classification of new objects.

        Parameters
        ----------
        filename : str
            Where the trained MLP is stored.
        config : ModelConfig
            Includes (in order): input_size, output_size, n_neurons, n_hidden.
            Also includes normalization means and standard deviations.

        Returns
        ----------
        torch.nn.Module
            The pre-trained MLP object.
        """
        config = ModelConfig.load(config_filename)
        model = SuperphotClassifier.create(config)  # set up empty multi-layer perceptron
        model.load_state_dict(torch.load(filename))  # load trained state dict to the MLP
        return model, config.normalization_means, config.normalization_stddevs
