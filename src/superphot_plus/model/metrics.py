from typing import List


class ModelMetrics:
    """Class containing the training and validation metrics."""

    train_acc: List[float] = []
    val_acc: List[float] = []
    train_loss: List[float] = []
    val_loss: List[float] = []

    epoch_mins: List[int] = []
    epoch_secs: List[int] = []
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

    def append(self, train_loss, train_acc, val_loss, val_acc, epoch_mins, epoch_secs):
        """Appends training information for an epoch.

        Parameters
        ----------
        train_loss: float
            The epoch training loss.
        train_acc: float
            The epoch training accuracy.
        val_loss: float
            The epoch validation loss.
        val_acc: float
            The epoch validation accuracy.
        epoch_mins: int
            The number of minutes spent by the epoch.
        epoch_secs: int
            The number of seconds spent by the epoch.
        """
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
