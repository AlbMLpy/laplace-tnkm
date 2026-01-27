import sys

import pytest

import jax
import numpy as np
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.append('./')

from source.evaluation import ecp, wcpi, rce, rmse, pll, nll, norm_frob


@pytest.fixture
def gaussian_data():
    """
    Well-specified regression setting:
    y ~ N(0,1), predictive samples ~ same distribution.
    """
    rng = np.random.default_rng(0)
    N, S = 2000, 1000
    y_true = rng.normal(size=(N,))
    y_dist = rng.normal(size=(N, S))
    return y_true, y_dist

@pytest.fixture
def gaussian_regression():
    """
    y_true ~ N(0, sigma**2), predictions are well-specified.
    """
    rng = np.random.default_rng(0)
    N = 2000
    sigma2 = 1.5

    y_true = rng.normal(scale=np.sqrt(sigma2), size=(N,))
    y_pred = y_true + rng.normal(scale=0.1, size=(N,))
    var_pred = np.full((N,), sigma2)

    return (
        jnp.array(y_pred),
        jnp.array(var_pred),
        jnp.array(y_true),
    )

# ---------------- WCPI ----------------

def test_wcpi_positive(gaussian_data):
    _, y_dist = gaussian_data
    width = wcpi(y_dist, alpha=0.9)
    assert width > 0

def test_wcpi_monotonic(gaussian_data):
    """ Test that interval width increases with alpha. """
    _, y_dist = gaussian_data
    w80 = wcpi(y_dist, alpha=0.8)
    w90 = wcpi(y_dist, alpha=0.9)
    w95 = wcpi(y_dist, alpha=0.95)
    assert w80 < w90 < w95

# ---------------- RCE ----------------

def test_rce_well_calibrated_small(gaussian_data):
    """ Test global calibration. """
    y_true, y_dist = gaussian_data
    err = rce(y_true, y_dist)
    assert err < 0.05

# ---------------- ECP ----------------

def test_ecp_well_calibrated(gaussian_data):
    """ Test correct empirical coverage. """
    y_true, y_dist = gaussian_data
    alpha = 0.9
    cov = ecp(y_true, y_dist, alpha)
    assert jnp.isclose(cov, alpha, atol=0.03)

def test_overconfident_model():
    """
    Predictive variance too small -> undercoverage -> overconfident.
    """
    rng = np.random.default_rng(1)
    N, S = 2000, 500
    y_true = rng.normal(size=N)
    y_dist = 0.2 * rng.normal(size=(N, S))  # too narrow
    cov = ecp(y_true, y_dist, alpha=0.9)
    assert cov < 0.9

def test_underconfident_model():
    """
    Predictive variance too large -> overcoverage -> underconfident
    """
    rng = np.random.default_rng(2)
    N, S = 2000, 500
    y_true = rng.normal(size=N)
    y_dist = 3.0 * rng.normal(size=(N, S))  # too wide
    cov = ecp(y_true, y_dist, alpha=0.9)
    assert cov > 0.9

def test_shape_robustness(gaussian_data):
    """ Test that y_true works as (N,) or (N,1). """
    y_true, y_dist = gaussian_data
    y_true_col = y_true[:, None]

    cov1 = ecp(y_true, y_dist, 0.9)
    cov2 = ecp(y_true_col, y_dist, 0.9)

    assert jnp.isclose(cov1, cov2)

def test_invalid_alpha():
    rng = np.random.default_rng(0)
    y_true = rng.normal(size=100)
    y_dist = rng.normal(size=(100, 50))
    with pytest.raises(ValueError):
        ecp(y_true, y_dist, alpha=1.2)

# ---------------- RMSE ----------------

def test_rmse_zero():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0

def test_rmse_known_value():
    y1 = np.array([0.0, 0.0])
    y2 = np.array([3.0, 4.0])
    assert np.isclose(rmse(y1, y2), 5.0 / np.sqrt(2))

# ---------------- PLL / NLL ----------------

def test_pll_higher_for_better_mean(gaussian_regression):
    y_pred, var_pred, y_true = gaussian_regression
    pll_good = pll(y_pred, var_pred, y_true)
    pll_bad = pll(y_pred + 3.0, var_pred, y_true)
    assert pll_good > pll_bad

def test_nll_is_negative_pll(gaussian_regression):
    y_pred, var_pred, y_true = gaussian_regression
    assert np.isclose(
        nll(y_pred, var_pred, y_true),
        -pll(y_pred, var_pred, y_true),
    )

# ---------------- Frobenius norm ----------------

def test_norm_frob_scalar():
    x = jnp.array(3.0)
    assert norm_frob(x) == 3.0

def test_norm_frob_vector():
    x = jnp.array([3.0, 4.0])
    assert norm_frob(x) == 5.0

def test_norm_frob_matrix():
    x = jnp.eye(3)
    assert np.isclose(norm_frob(x), np.sqrt(3.0))
