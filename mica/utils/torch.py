import copy
import pathlib
import sys
from contextlib import AbstractContextManager
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from mica.logger import logger
from mica.settings import settings


class EarlyStopping:
    """
    Args:
        monitor: Quantity to be monitored.
        mode: One of {"min", "max"}. In min mode, stopping occurs when the quantity
              monitored has stopped decreasing; in "max" mode it will stop when the
              quantity monitored has stopped increasing.
        patience: Number of epochs with no improvement after which training will be stopped.
        threshold: An absolute threshold for the monitored quantity. Stops training when reached.
        min_delta: Minimum change in the monitored quantity to qualify as an improvement.
        restore_best_weights: Whether to restore model weights from the epoch with the best
                              value of the monitored quantity.
    """

    def __init__(self, monitor, mode="min", patience=None, threshold=None, min_delta=0, restore_best_weights=False):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.threshold = threshold
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_weights = None

        if self.patience is None and self.threshold is None:
            raise ValueError("Either 'patience' or 'threshold' must be specified")

    def __call__(self, history, model):
        score = history[self.monitor][-1]

        if self.best_score is None:
            self.best_score = score
            if self.restore_best_weights:
                self.best_weights = copy.deepcopy(model.state_dict())

        if self.threshold is not None:
            # Threshold-based stopping
            if (self.mode == "min" and score <= self.threshold) or (self.mode == "max" and score >= self.threshold):
                print(f"\nReached threshold: {score}")
                self.early_stop = True
        elif self.patience is not None:
            # Patience-based stopping
            if (self.mode == "min" and score > self.best_score - self.min_delta) or (
                self.mode == "max" and score < self.best_score + self.min_delta
            ):
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
                if self.restore_best_weights:
                    self.best_weights = copy.deepcopy(model.state_dict())

        return self.early_stop

    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class ProgressBar(AbstractContextManager):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def __init__(self, total_samples):
        self.loss = float("nan")
        self.acc = float("nan")
        self.val_loss = float("nan")
        self.val_acc = float("nan")
        self.pbar = tqdm.tqdm(
            bar_format="{n_fmt}/{total_fmt} [{bar:30}] {elapsed}<{remaining} {rate_fmt} | {desc}",
            disable=settings.DISABLE_PROGRESS_BAR,
            total=total_samples,
            unit="sample",
        )
        self.pbar.set_description_str(self.get_description())

    def get_description(self):
        return (
            f"loss: {self.loss:.3f} | "
            f"acc: {self.acc:.2f} | "
            f"val_loss: {self.val_loss:.3f} | "
            f"val_acc: {self.val_acc:.2f}"
        )

    def close(self):
        self.pbar.close()

    def update(self, n):
        self.pbar.set_description_str(self.get_description())
        self.pbar.update(n)


class Trainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        self.device: str = kwargs.get("device") or "cpu"

        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer

        self.history = {
            "train_acc": [],
            "train_loss": [],
            "val_acc": [],
            "val_loss": [],
        }

        self.early_stopping_criteria: List[EarlyStopping] = kwargs.get("early_stopping_criteria") or []

        # Configure checkpoint directory.
        checkpoint_path = kwargs.get("checkpoint_path")
        if isinstance(checkpoint_path, pathlib.Path):
            self.checkpoint_path = checkpoint_path
        elif isinstance(checkpoint_path, str):
            self.checkpoint_path = pathlib.Path(checkpoint_path)
        else:
            self.checkpoint_path = pathlib.Path("./output/checkpoints")
        self.checkpoint_path.mkdir(exist_ok=True, parents=True)

        self.best_params: Dict[str, Any] = {
            "model": None,
            "optimizer": None,
        }

    def train(self, train_loader: DataLoader, val_loader: DataLoader, **kwargs) -> None:
        epochs: int = kwargs["epochs"]
        total_train_samples = len(train_loader.dataset)

        for epoch in range(epochs):
            if epoch % 500 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}")

            pbar = ProgressBar(total_train_samples)

            train_loss, train_acc = self.train_epoch(train_loader, pbar=pbar)
            val_loss, val_acc = self.train_epoch(val_loader, pbar=pbar, is_validation=True)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            # Check if this is the best model
            if val_acc >= max(self.history["val_acc"]):
                # self.save_checkpoint(is_best=True)
                self.best_params["model"] = self.model.state_dict()
                self.best_params["optimizer"] = self.optimizer.state_dict()

            # Save regular checkpoint
            # self.save_checkpoint()

            pbar.close()

            # Check stop criteria
            if self.early_stop():
                print("Stop criteria met. Ending training.")
                break

    def train_epoch(self, dataloader, pbar: Optional[ProgressBar] = None, is_validation: bool = False):
        if is_validation:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            if is_validation:
                ctx = torch.no_grad
            else:
                ctx = torch.enable_grad

            with ctx():
                data = batch["data"].to(self.device)
                target = batch["label"].to(self.device)

                if not is_validation:
                    self.optimizer.zero_grad()

                output = self.model(data.unsqueeze(1))
                loss = self.criterion(output, target)

                if not is_validation:
                    loss.backward()
                    self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                if pbar:
                    if is_validation:
                        pbar.val_loss = total_loss / total
                        pbar.val_acc = correct / total
                        pbar.update(0)
                    else:
                        pbar.loss = total_loss / total
                        pbar.acc = correct / total
                        pbar.update(data.size(0))

        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = correct / total

        return avg_loss, accuracy

    def load_best_params(self):
        self.model.load_state_dict(self.best_params["model"])
        self.optimizer.load_state_dict(self.best_params["optimizer"])

    def save_checkpoint(self, is_best=False):
        checkpoint = {
            # 'epoch': epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            # 'best_val_loss': self.best_val_loss,
            # 'train_losses': self.train_losses,
            # 'val_losses': self.val_losses,
            # 'train_accuracies': self.train_accuracies,
            # 'val_accuracies': self.val_accuracies
        }

        filename = f"last.pth"
        torch.save(checkpoint, self.checkpoint_path / filename)

        if is_best:
            best_filename = "best.pth"
            torch.save(checkpoint, self.checkpoint_path / best_filename)

    def load_checkpoint(self, checkpoint_path: pathlib.Path):
        checkpoint = torch.load(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # self.start_epoch = checkpoint['epoch'] + 1
        # self.best_val_loss = checkpoint['best_val_loss']
        # self.train_losses = checkpoint['train_losses']
        # self.val_losses = checkpoint['val_losses']
        # self.train_accuracies = checkpoint['train_accuracies']
        # self.val_accuracies = checkpoint['val_accuracies']

    def early_stop(self) -> bool:
        # Early stopping checks
        for criterion in self.early_stopping_criteria:
            if criterion(self.history, self.model):
                if criterion.restore_best_weights:
                    criterion.restore_weights(self.model)
                return True

        return False

    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                data = batch["data"].to(self.device)
                target = batch["label"].to(self.device)

                outputs = self.model(data.unsqueeze(1))
                _, predicted = torch.max(outputs, 1)

                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())

        return np.array(all_predictions), np.array(all_labels)

    def evaluate(self, test_loader: DataLoader):
        predictions, true_labels = self.predict(test_loader)

        return {
            "acc": accuracy_score(true_labels, predictions),
            "clf_report": classification_report(true_labels, predictions, output_dict=True),
            "confusion_matrix": confusion_matrix(true_labels, predictions).tolist(),
            "y_pred": predictions.tolist(),
            "y_true": true_labels.tolist(),
        }


def dataset_to_numpy(dataset: Dataset):
    # Assuming your dataset is called 'your_dataset'
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

    data = []
    labels = []

    for batch in dataloader:
        data.append(batch["data"].cpu().numpy())
        labels.append(batch["label"].cpu().numpy())

    # Convert to numpy arrays
    data_np = np.array(data).flatten()
    labels_np = np.array(labels).flatten()

    # Reshape the data
    n_trials = len(dataset)
    data_reshaped = data_np.reshape(n_trials, 8, 750)

    return data_reshaped, labels_np
