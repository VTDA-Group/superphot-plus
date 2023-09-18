from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelMetrics:
    """Class containing the training and validation metrics."""

    train_acc: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)

    epoch_mins: List[int] = field(default_factory=list)
    epoch_secs: List[int] = field(default_factory=list)
    curr_epoch: int = 0

    def get_values(self):
        """Returns the training and validation accuracies and losses.

        Returns
        -------
        tuple
            A tuple containing the training accuracy and loss, and
            validation accuracy and loss, respectively.
        """
        return self.train_acc, self.train_loss, self.val_acc, self.val_loss

    def append(self, train_metrics, val_metrics, epoch_time):
        """Appends training information for an epoch.

        Parameters
        ----------
        train_metrics: tuple
            The epoch training loss and accuracy.
        val_metrics: tuple
            The epoch validation loss and accuracy.
        epoch_time: tuple
            The number of minutes and seconds spent by the epoch.
        """
        train_loss, train_acc = train_metrics
        val_loss, val_acc = val_metrics
        epoch_mins, epoch_secs = epoch_time

        self.curr_epoch += 1

        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.epoch_mins.append(epoch_mins)
        self.epoch_secs.append(epoch_secs)

    def print_last(self):
        """Prints the metrics for the last epoch."""
        epoch_mins, epoch_secs, train_loss, train_acc, val_loss, val_acc = (
            self.epoch_mins[-1],
            self.epoch_secs[-1],
            self.train_loss[-1],
            self.train_acc[-1],
            self.val_loss[-1],
            self.val_acc[-1],
        )
        print(f"Epoch: {self.curr_epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc*100:.2f}%")


@dataclass
class RegressorMetrics:
    """Class containing the training and validation metrics."""

    train_loss: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)

    epoch_mins: List[int] = field(default_factory=list)
    epoch_secs: List[int] = field(default_factory=list)
    curr_epoch: int = 0

    def get_values(self):
        return self.train_loss, self.val_loss

    def append(self, train_loss, val_loss, epoch_time):
        epoch_mins, epoch_secs = epoch_time

        self.curr_epoch += 1

        self.train_loss.append(train_loss)
        self.val_loss.append(val_loss)
        self.epoch_mins.append(epoch_mins)
        self.epoch_secs.append(epoch_secs)

    def print_last(self):
        """Prints the metrics for the last epoch."""
        epoch_mins, epoch_secs, train_loss, val_loss = (
            self.epoch_mins[-1],
            self.epoch_secs[-1],
            self.train_loss[-1],
            self.val_loss[-1],
        )
        print(f"Epoch: {self.curr_epoch:02} | Epoch Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f}")
        print(f"\t Val. Loss: {val_loss:.3f}")
