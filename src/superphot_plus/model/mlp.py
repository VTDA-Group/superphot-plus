import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.nn.modules.loss import _Loss

from superphot_plus.constants import HIDDEN_DROPOUT_FRAC, INPUT_DROPOUT_FRAC
from superphot_plus.model.config import ModelConfig


class SuperphotMlp(nn.Module):
    """The Multi-Layer Perceptron.

    Parameters
    ----------
    config : ModelConfig
        The neural network specification.
    criterion : _Loss
        The loss function.
    """

    def __init__(self, config: ModelConfig, criterion: _Loss):
        super().__init__()
        self.config = config
        self.init_nn()
        self.criterion = criterion
        self.best_model = None  # State dictionary

    def init_nn(self):
        """Builds network architecture."""
        input_dim = self.config.input_dim
        output_dim = self.config.output_dim
        n_neurons = self.config.neurons_per_layer
        num_hidden_layers = self.config.num_hidden_layers
        learning_rate = self.config.learning_rate

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

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

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

    def save(self, models_dir, name="best-model"):
        """Stores the trained model and respective configuration.

        Parameters
        ----------
        models_dir : str, optional
            Where to store pretrained models and their configurations.
        name : str, optional
            The name of the model to be saved. Defaults to "best-model".
        """
        file_prefix = os.path.join(models_dir, name)

        # Save configuration to disk
        self.config.write_to_file(f"{file_prefix}.yaml")

        # Save Pytorch model to disk
        torch.save(self.best_model, f"{file_prefix}.pt")
