"""This module implements the Multi-Layer Perceptron (MLP) model for classification."""
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from superphot_plus.constants import EPOCHS, HIDDEN_DROPOUT_FRAC, INPUT_DROPOUT_FRAC
from superphot_plus.format_data_ztf import normalize_features
from superphot_plus.config import SuperphotConfig
from superphot_plus.model.metrics import ModelMetrics
from superphot_plus.utils import (
    create_dataset,
    epoch_time,
    save_test_probabilities,
    calculate_accuracy,
)


class SuperphotMLP(nn.Module):
    """The Multi-Layer Perceptron.

    Parameters
    ----------
    config : ModelConfig
        The MLP architecture configuration.
    """

    def __init__(self, config: SuperphotConfig):
        super().__init__()

        # Initialize MLP architecture
        self.config = config

        n_neurons = config.neurons_per_layer
        self.input_fc = nn.Linear(config.input_dim, n_neurons)

        assert config.num_hidden_layers >= 1

        self.hidden_layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.dropouts.append(nn.Dropout(INPUT_DROPOUT_FRAC))

        for _ in range(config.num_hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(n_neurons, n_neurons))
        for _ in range(config.num_hidden_layers):
            self.dropouts.append(nn.Dropout(HIDDEN_DROPOUT_FRAC))

        self.output_fc = nn.Linear(n_neurons, config.output_dim)

        # Optimizer and criterion
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # Model state dictionary
        self.best_model = None

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

    def train_and_validate(
        self,
        train_data,
        num_epochs=EPOCHS,
        rng_seed=None,
    ):
        """
        Run the MLP initialization and training.

        Closely follows the demo
        https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

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
            A tuple containing arrays of metrics for each epoch
            (training accuracies and losses, validation accuracies and losses).
        """
        if rng_seed is not None:
            random.seed(rng_seed)
            np.random.seed(rng_seed)
            torch.manual_seed(rng_seed)
            torch.cuda.manual_seed(rng_seed)
            torch.backends.cudnn.deterministic = True

        train_dataset, valid_dataset = train_data

        train_iterator = DataLoader(
            dataset=train_dataset, shuffle=True,
            batch_size=self.config.batch_size,
            pin_memory=True
        )
        valid_iterator = DataLoader(
            dataset=valid_dataset,
            batch_size=self.config.batch_size,
            pin_memory=True
        )

        metrics = ModelMetrics()

        best_model = None
        best_val_loss = float("inf")

        for epoch in np.arange(0, num_epochs):
            
            start_time = time.monotonic()

            train_loss, train_acc = self.train_epoch(train_iterator)
            val_loss, val_acc = self.evaluate_epoch(valid_iterator)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.state_dict()

            end_time = time.monotonic()

            # Store metrics for the current epoch
            metrics.append(
                train_metrics=(train_loss, train_acc),
                val_metrics=(val_loss, val_acc),
                epoch_time=epoch_time(start_time, end_time),
            )

            if epoch % 50 == 0:
                metrics.print_last()

        # Save best model state
        self.best_model = best_model
        self.load_state_dict(best_model)

        # Store best validation loss
        self.config.set_best_val_loss(best_val_loss)

        return metrics.get_values()

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
            #x = x.to(self.config.device)
            #y = y.to(self.config.device)

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

    def evaluate(self, test_data):
        """Runs model over a group of test samples.

        Parameters
        ----------
        test_data : TestData
            The data to evaluate the model. Consists of test features,
            test classes, test names and a list of grouped indices, respectively.

        Returns
        -------
        tuple
            A tuple containing the labels, names, predicted labels
            and maximum probabilities.
        """
        test_features, test_classes, test_names = test_data

        # Write output file header
        with open(self.config.probs_fn, "w+", encoding="utf-8") as probs_file:
            probs_file.write("Name,Label,pSNIa,pSNII,pSNIIn,pSLSNI,pSNIbc\n")

        labels, pred_labels, max_probs, names, probs_avgs = [], [], [], [], []

        for test_name in test_names:
            group_idx_set = (test_names == test_name)
            test_dataset = create_dataset(
                test_features[group_idx_set],
                test_classes[group_idx_set],
            )

            test_iterator = DataLoader(
                dataset=test_dataset, batch_size=self.config.batch_size
            )

            _, labels_indiv, probs = self.get_predictions(
                test_iterator
            )
            probs_avg = np.mean(probs.numpy(), axis=0)

            labels_indiv = labels_indiv.numpy()

            pred_labels.append(np.argmax(probs_avg))
            max_probs.append(np.amax(probs_avg))
            labels.append(labels_indiv[0])
            names.append(test_name)
            probs_avgs.append(probs_avg)

        save_test_probabilities(
            names,
            np.array(probs_avgs),
            self.config.probs_fn,
            true_labels=labels,
        )
        return (
            np.array(labels).astype(int),
            np.array(names),
            np.array(pred_labels).astype(int),
            np.array(max_probs).astype(float),
        )

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

        with torch.no_grad():
            for x, y in iterator:
                x = x.to(self.config.device)

                y_pred, _ = self(x)

                y_prob = F.softmax(y_pred, dim=-1)

                images.append(x.cpu())
                labels.append(y.cpu())
                probs.append(y_prob.cpu())

        images = torch.cat(images, dim=0)
        labels = torch.cat(labels, dim=0)
        probs = torch.cat(probs, dim=0)

        return images, labels, probs

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


    def save(self, models_dir, suffix=""):
        """Stores the trained model and respective configuration.

        Parameters
        ----------
        models_dir : str, optional
            Where to store pretrained models and their configurations.
        """
        file_prefix = os.path.join(models_dir, f"best-model-{suffix}")

        # Save configuration to disk
        self.config.write_to_file(f"{file_prefix}.yaml")

        # Save Pytorch model to disk
        torch.save(self.best_model, f"{file_prefix}.pt")

        
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
        model = cls(config)
        model.criterion = model.criterion.to(config.device)
        model = model.to(config.device)
        return model

    @classmethod
    def load(cls, filename, config_filename):
        """Load a trained MLP for subsequent classification of new objects.

        Parameters
        ----------
        filename : str
            The path to the pre-trained model.
        config_filename : str
            The file that includes the model training configuration.

        Returns
        ----------
        tuple
            The pre-trained classifier object and the respective model config.
        """
        config = SuperphotConfig.from_file(config_filename)
        model = SuperphotMLP.create(config)  # set up empty multi-layer perceptron
        model.load_state_dict(torch.load(filename))  # load trained state dict to the MLP
        return model, config
