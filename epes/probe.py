from __future__ import annotations
import numpy as np
import pandas as pd

def make_probe_df(
    X_train: pd.DataFrame,
    feature: str,
    domain_min: float,
    domain_max: float,
    num_points: int = 100
) -> pd.DataFrame:
    """
    Build a probe DataFrame where `feature` varies across [domain_min, domain_max]
    and all other columns are fixed to their training means.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features used to derive reference (mean) values.
    feature : str
        Column to vary.
    domain_min : float
        Lower bound of the probe range.
    domain_max : float
        Upper bound of the probe range.
    num_points : int, optional
        Number of probe points (default 100).

    Returns
    -------
    pd.DataFrame
        Probe dataset with same columns as X_train.
    """
    if feature not in X_train.columns:
        raise ValueError(f"feature '{feature}' not found in X_train columns.")
    if num_points <= 1:
        raise ValueError("num_points must be > 1.")
    if domain_max <= domain_min:
        raise ValueError("domain_max must be > domain_min.")

    probe_vals = np.linspace(domain_min, domain_max, num_points)
    fixed_means = X_train.drop(columns=[feature]).mean(numeric_only=True).to_dict()

    probe_df = pd.DataFrame({feature: probe_vals})
    for col in X_train.columns:
        if col == feature:
            continue
        # If column is non-numeric, fallback to mode if available
        if col in fixed_means:
            probe_df[col] = fixed_means[col]
        else:
            mode_series = X_train[col].mode()
            if mode_series.empty:
                raise ValueError(f"Cannot determine reference value for column '{col}'.")
            probe_df[col] = mode_series.iloc[0]

    # Ensure column order matches X_train
    probe_df = probe_df[X_train.columns]
    return probe_df
