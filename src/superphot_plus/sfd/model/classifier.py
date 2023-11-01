"""This module implements the Multi-Layer Perceptron (MLP) model for classification."""
import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from superphot_plus.constants import EPOCHS, HIDDEN_DROPOUT_FRAC, INPUT_DROPOUT_FRAC
from superphot_plus.file_paths import PROBS_FILE
from superphot_plus.file_utils import get_posterior_samples
from superphot_plus.format_data_ztf import normalize_features
from superphot_plus.model.config import ModelConfig
from superphot_plus.model.metrics import ModelMetrics
from superphot_plus.utils import (
    adjust_log_dists,
    calculate_accuracy,
    create_dataset,
    epoch_time,
    save_test_probabilities,
)


class SuperphotClassifier(nn.Module):
    """The Multi-Layer Perceptron.

    Parameters
    ----------
    config : ModelConfig
        The MLP architecture configuration.
    """

    def __init__(self, config: ModelConfig):
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

        train_iterator = DataLoader(dataset=train_dataset, shuffle=True, batch_size=self.config.batch_size)
        valid_iterator = DataLoader(dataset=valid_dataset, batch_size=self.config.batch_size)

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

    def evaluate(self, test_data, probs_csv_path=PROBS_FILE):
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
        """
        test_features, test_classes, test_names, test_group_idxs = test_data

        # Write output file header
        with open(probs_csv_path, "w+", encoding="utf-8") as probs_file:
            probs_file.write("Name,Label,pSNIa,pSNII,pSNIIn,pSLSNI,pSNIbc\n")

        labels, pred_labels, max_probs, names = [], [], [], []

        for group_idx_set in test_group_idxs:
            test_dataset = create_dataset(
                test_features[group_idx_set],
                test_classes[group_idx_set],
                group_idx_set,
            )

            test_iterator = DataLoader(dataset=test_dataset, batch_size=self.config.batch_size)

            _, labels_indiv, indx_indiv, probs = self.get_predictions(test_iterator)
            probs_avg = np.mean(probs.numpy(), axis=0)

            labels_indiv = labels_indiv.numpy()

            save_test_probabilities(
                test_names[indx_indiv.numpy().astype(int)[0]],
                probs_avg,
                labels_indiv[0],
                save_file=probs_csv_path,
            )

            pred_labels.append(np.argmax(probs_avg))
            max_probs.append(np.amax(probs_avg))
            labels.append(labels_indiv[0])
            names.append(test_names[indx_indiv.numpy().astype(int)[0]])

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
        post_features = get_posterior_samples(obj_name, fits_dir, sampler)

        chisq = np.mean(post_features[:, -1])
        if np.abs(chisq) > 10:  # probably not a SN
            print("OBJECT LIKELY NOT A SN")

        # normalize the log distributions
        post_features = adjust_log_dists(post_features)
        probs = self.classify_from_fit_params(post_features)
        probs_avg = np.mean(probs, axis=0)
        return probs_avg

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

    def return_new_classifications(self, test_csv, fit_dir, save_file, output_dir=None, include_labels=False):
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

        with open(filepath, "w+", encoding="utf-8") as pf:
            pf.write("Name,Label,pSNIa,pSNII,pSNIIn,pSLSNI,pSNIbc\n")

        with open(test_csv, "r", encoding="utf-8") as tc:
            csv_reader = csv.reader(tc, delimiter=",")
            next(csv_reader)
            for _, row in enumerate(csv_reader):
                try:
                    test_name = row[0]
                except:
                    print(row, "skipped")
                    continue

                label = None

                if include_labels:
                    label = row[1]

                probs_avg = self.classify_single_light_curve(test_name, fit_dir)
                save_test_probabilities(test_name, probs_avg, label, output_dir, save_file)

    def save(self, models_dir):
        """Stores the trained model and respective configuration.

        Parameters
        ----------
        models_dir : str, optional
            Where to store pretrained models and their configurations.
        """
        file_prefix = os.path.join(models_dir, "best-model")

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
        config = ModelConfig.from_file(config_filename)
        model = SuperphotClassifier.create(config)  # set up empty multi-layer perceptron
        model.load_state_dict(torch.load(filename))  # load trained state dict to the MLP
        return model, config
