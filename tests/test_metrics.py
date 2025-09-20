import numpy as np
from epes.metrics import epe_metrics

def test_epe_metrics_simple():
    # f(x) = 1 + x; g(x) = 1 + 2x
    f = np.array([1.0, 1.0])   # a0=1, a1=1
    g = np.array([1.0, 2.0])   # a0=1, a1=2
    phi, dphi, phi_star = epe_metrics(f, g, alpha=0.7, beta=0.3, p=2)

    # Static difference: |[1,1] - [1,2]|_2 = |[0,-1]|_2 = 1
    assert np.isclose(phi, 1.0)
    # Derivatives: f'=(1), g'=(2) -> |1-2|=1
    assert np.isclose(dphi, 1.0)
    # Fusion: 0.7*1 + 0.3*1 = 1
    assert np.isclose(phi_star, 1.0)
