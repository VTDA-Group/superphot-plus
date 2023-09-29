import os

import matplotlib.pyplot as plt
import numpy as np


def plot_model_metrics(metrics, num_epochs, plot_name, metrics_dir):
    """Plots training and validation results and exports them to files.

    Parameters
    ----------
    metrics: tuple
        Train and validation accuracies and losses.
    num_epochs: int
        The total number of epochs.
    plot_name: str
        The name for the plot figure files.
    metrics_dir: str
        Where to store the plot figures.
    """
    train_loss, val_loss = metrics

    # Plot loss
    plt.plot(np.arange(0, num_epochs), train_loss, label="Training")
    plt.plot(np.arange(0, num_epochs), val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(metrics_dir, f"loss_{plot_name}.pdf"), bbox_inches="tight")
    plt.close()
