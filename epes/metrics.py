from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray
from numpy.linalg import norm
from .utils import poly_deriv_coeffs

def _pad_to_same_length(a: NDArray[np.float64], b: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = max(a.size, b.size)
    a_pad = np.pad(a, (0, n - a.size), mode="constant", constant_values=0.0)
    b_pad = np.pad(b, (0, n - b.size), mode="constant", constant_values=0.0)
    return a_pad, b_pad

def epe_metrics(
    coeff_f: ArrayLike,
    coeff_g: ArrayLike,
    alpha: float = 0.7,
    beta: float = 0.3,
    p: int | float = 2
) -> tuple[float, float, float]:
    """
    Compute Epe metrics between two fitted polynomials f and g.

    ϝ  (phi):     static divergence of coefficient vectors.
    δϝ (dphi):    divergence of derivative coefficient vectors.
    ϝ* (phi_star): fusion metric alpha*ϝ + beta*δϝ.

    Parameters
    ----------
    coeff_f : array-like
        Coefficients of f in increasing order.
    coeff_g : array-like
        Coefficients of g in increasing order.
    alpha : float, optional
        Weight for ϝ in fusion metric.
    beta : float, optional
        Weight for δϝ in fusion metric.
    p : int|float, optional
        Norm order for vector norms (default 2).

    Returns
    -------
    (phi, dphi, phi_star) : tuple of floats
        Static divergence, rate divergence, and fusion metric.

    Notes
    -----
    - Coefficient vectors are padded to the same length.
    - Derivative vectors are also padded before norm computation.
    """
    f = np.asarray(coeff_f, dtype=float)
    g = np.asarray(coeff_g, dtype=float)

    f, g = _pad_to_same_length(f, g)
    f1 = poly_deriv_coeffs(f)
    g1 = poly_deriv_coeffs(g)
    f1, g1 = _pad_to_same_length(f1, g1)

    # Use p-norm. For p=2 we use numpy.linalg.norm; for general p, use np.linalg.norm with ord=p on flat vectors.
    phi = float(norm(f - g, ord=p))
    dphi = float(norm(f1 - g1, ord=p))
    phi_star = alpha * phi + beta * dphi
    return phi, dphi, phi_star
