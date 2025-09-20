"""
Epe Framework (epes): coefficient-space auditing metrics (ϝ, δϝ, ϝ*).
"""
from .metrics import epe_metrics
from .probe import make_probe_df
from .plots import slice_plot
from .utils import fit_poly, poly_deriv_coeffs

__all__ = [
    "epe_metrics",
    "make_probe_df",
    "slice_plot",
    "fit_poly",
    "poly_deriv_coeffs",
]
