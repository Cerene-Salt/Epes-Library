from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike, NDArray

def fit_poly(x: ArrayLike, y: ArrayLike, deg: int) -> NDArray[np.float64]:
    """
    Fit a polynomial p(x) of degree `deg` to points (x, y) via least squares.

    Parameters
    ----------
    x : array-like
        1D array of x-values.
    y : array-like
        1D array of y-values.
    deg : int
        Polynomial degree (>=0).

    Returns
    -------
    np.ndarray
        Coefficients in increasing order [a0, a1, ..., adeg].
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D arrays.")
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length.")
    if deg < 0:
        raise ValueError("deg must be >= 0.")

    V = np.vander(x_arr, N=deg + 1, increasing=True)
    coeffs, *_ = np.linalg.lstsq(V, y_arr, rcond=None)
    return coeffs.astype(float)

def poly_deriv_coeffs(coeffs: ArrayLike) -> NDArray[np.float64]:
    """
    Compute coefficients of the derivative p'(x) given p(x) coefficients.

    Parameters
    ----------
    coeffs : array-like
        Coefficients in increasing order [a0, a1, ..., an].

    Returns
    -------
    np.ndarray
        Derivative coefficients [a1*1, a2*2, ..., an*n].
    """
    c = np.asarray(coeffs, dtype=float)
    if c.ndim != 1:
        raise ValueError("coeffs must be 1D.")
    if c.size == 0:
        return np.array([], dtype=float)
    return np.array([i * c[i] for i in range(1, c.size)], dtype=float)
