from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

def slice_plot(
    x_vals: ArrayLike,
    y_hat_a: ArrayLike,
    y_hat_b: ArrayLike,
    feature: str,
    labels: tuple[str, str] = ("Model A", "Model B")
) -> None:
    """
    Plot 1D slice of predictions for two models across a single feature.

    Parameters
    ----------
    x_vals : array-like
        Values of the audited feature.
    y_hat_a : array-like
        Predictions from model A (same length as x_vals).
    y_hat_b : array-like
        Predictions from model B (same length as x_vals).
    feature : str
        Feature name for labeling.
    labels : tuple[str, str], optional
        Legend labels for the two models.
    """
    x = np.asarray(x_vals)
    ya = np.asarray(y_hat_a)
    yb = np.asarray(y_hat_b)
    if not (x.size == ya.size == yb.size):
        raise ValueError("x_vals, y_hat_a, and y_hat_b must have equal lengths.")

    plt.figure(figsize=(10, 5))
    plt.plot(x, ya, label=labels[0], color="blue")
    plt.plot(x, yb, label=labels[1], color="orange")
    plt.xlabel(feature)
    plt.ylabel("Prediction")
    plt.title(f"Comparison of Models along {feature}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
