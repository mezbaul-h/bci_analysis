import numpy as np
from matplotlib import pyplot as plt


def draw_loss_acc_graph(training_losses, training_acc, validation_losses, validation_acc, filename=None):
    """
    Plots training and validation loss and accuracy on a single figure.

    Args:
        training_losses: List of training losses.
        training_acc: List of training accuracies.
        validation_losses: List of validation losses.
        validation_acc: List of validation accuracies.
    """

    # Create a figure with two subplots (1 row, 2 columns)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot training and validation loss
    axs[0].plot(training_losses, label="Training Loss")
    axs[0].plot(validation_losses, label="Validation Loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot training and validation accuracy
    axs[1].plot(np.array(training_acc) * 100, label="Training Accuracy")
    axs[1].plot(np.array(validation_acc) * 100, label="Validation Accuracy")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.tight_layout()
    # plt.show()

    if not filename:
        filename = "loss_acc_graph.png"

    plt.savefig(filename, dpi=300, bbox_inches="tight")

    # Close the plot to free up memory
    plt.close()
