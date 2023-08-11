"""This module implements the Multi-Layer Perceptron (MLP) model for
classification."""

from dataclasses import dataclass
import os
import random
import time

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from superphot_plus.constants import (
    BATCH_SIZE,
    EPOCHS,
    HIDDEN_DROPOUT_FRAC,
    INPUT_DROPOUT_FRAC,
    LEARNING_RATE,
    SEED,
)
from superphot_plus.file_paths import METRICS_DIR, MODELS_DIR
from superphot_plus.utils import (
    calculate_accuracy,
    create_dataset,
    epoch_time,
    save_test_probabilities,
)
from superphot_plus.plotting import plot_model_metrics


@dataclass
class ModelConfig:
    """Class that holds the MLP configuration."""

    input_dim: int
    output_dim: int
    neurons_per_layer: int
    num_hidden_layers: int

    device: torch.device = torch.device("cpu")

    def __iter__(self):
        return iter((self.input_dim, self.output_dim, self.neurons_per_layer, self.num_hidden_layers))


@dataclass
class ModelData:
    """Class that holds the MLP data to train / test / validate."""

    train_data: TensorDataset
    valid_data: TensorDataset
    test_sample_features: np.ndarray
    test_sample_classes: np.ndarray
    test_sample_names: np.ndarray
    test_group_idxs: list[int]

    def __iter__(self):
        return iter(
            (
                self.train_data,
                self.valid_data,
                self.test_sample_features,
                self.test_sample_classes,
                self.test_sample_names,
                self.test_group_idxs,
            )
        )


@dataclass
class ModelMetrics:
    """Class containing the training results."""

    train_acc: list
    val_acc: list
    train_loss: list
    val_loss: list
    num_epochs: int

    def __iter__(self):
        return iter(
            (
                self.train_acc,
                self.val_acc,
                self.train_loss,
                self.val_loss,
                self.num_epochs,
            )
        )


class MLP(nn.Module):
    """The Multi-Layer Perceptron. Sets the number of layers and nodes
    per layer.

    Parameters
    ----------
    config : ModelConfig
        The MLP architecture configuration.
    data : ModelData
        The training, test and validation data.
    """

    def __init__(self, config: ModelConfig, data: ModelData):
        super().__init__()

        # Initialize MLP architecture
        self.config = config

        input_dim, output_dim, neurons_per_layer, num_hidden_layers = config

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
        self.optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        self.criterion = nn.CrossEntropyLoss()

        # Training, test and validation data
        self.data = data

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
        num_epochs=EPOCHS,
        plot_metrics=False,
        metrics_dir=METRICS_DIR,
        models_dir=MODELS_DIR,
    ):
        """
        Run the MLP initialization and training.

        Closely follows the demo
        https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/1_mlp.ipynb

        Parameters
        ----------
        num_epochs : int, optional
            The number of epochs. Defaults to EPOCHS.
        plot_metrics : bool, optional
            Whether to plot metrics. Defaults to False.
        metrics_dir : str, optional
            Where to store metrics.
        models_dir : str, optional
            Where to store models.

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
            train_data,
            valid_data,
            test_sample_features,
            test_sample_classes,
            test_sample_names,
            test_group_idxs,
        ) = self.data

        train_iterator = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
        valid_iterator = DataLoader(valid_data, batch_size=BATCH_SIZE)

        best_valid_loss = float("inf")

        train_acc_arr = []
        train_loss_arr = []
        val_acc_arr = []
        val_loss_arr = []

        for epoch in np.arange(0, num_epochs):
            start_time = time.monotonic()

            train_loss, train_acc = self.train_epoch(train_iterator)
            valid_loss, valid_acc = self.evaluate_epoch(valid_iterator)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(
                    self.state_dict(),
                    os.path.join(models_dir, f"superphot-model-{test_sample_names[0]}.pt"),
                )

            end_time = time.monotonic()

            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            if epoch % 5 == 0:
                print(f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
                print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
                print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")

            train_loss_arr.append(train_loss)
            train_acc_arr.append(train_acc)
            val_loss_arr.append(valid_loss)
            val_acc_arr.append(valid_acc)

        self.load_state_dict(
            torch.load(os.path.join(models_dir, f"superphot-model-{test_sample_names[0]}.pt"))
        )

        labels, names, pred_labels, max_probs = self.test(
            test_sample_features, test_sample_classes, test_sample_names, test_group_idxs
        )

        if plot_metrics:
            plot_model_metrics(
                metrics=ModelMetrics(
                    train_acc=train_acc_arr,
                    val_acc=val_acc_arr,
                    train_loss=train_loss_arr,
                    val_loss=val_loss_arr,
                    num_epochs=num_epochs,
                ),
                plot_name=test_sample_names[0],
                metrics_dir=metrics_dir,
            )

        return (
            np.array(labels).astype(int),
            np.array(names),
            np.array(pred_labels).astype(int),
            np.array(max_probs).astype(float),
            best_valid_loss,
        )

    def test(self, test_features, test_classes, test_names, test_group_idxs):
        """Runs model over a group of test samples

        Parameters
        ----------
        test_features : np.ndarray
            The features array.
        test_classes : np.ndarray
            The classes array.
        test_names : np.ndarray
            The names array.
        test_group_idxs : np.ndarray
            The indices for each test set.

        Returns
        -------
        tuple
            A tuple containing the labels, names, predicted labels, maximum
            probabilities, and best validation loss.
        """
        labels, pred_labels, max_probs, names = [], [], [], []

        for group_idx_set in test_group_idxs:
            test_data = create_dataset(
                test_features[group_idx_set],
                test_classes[group_idx_set],
                group_idx_set,
            )
            test_iterator = DataLoader(test_data, batch_size=BATCH_SIZE)

            _, labels_indiv, indx_indiv, probs = self.get_predictions(test_iterator)
            probs_avg = np.mean(probs.numpy(), axis=0)

            save_test_probabilities(
                test_names[indx_indiv.numpy().astype(int)[0]],
                probs_avg,
                labels_indiv[0],
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

    def get_predictions_new(self, iterator):
        """Given a trained model, returns the test images, test labels, and
        prediction probabilities across all the test labels.

        Parameters
        ----------
        model : mlp.MLP
            The trained model.

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

    @classmethod
    def create(cls, config, data=None):
        """Creates an MLP instance, optimizer and respective criterion."""
        model = cls(config, data)
        model.criterion = model.criterion.to(config.device)
        model = model.to(config.device)
        return model
