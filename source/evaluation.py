import jax
import jax.numpy as jnp

from sklearn.metrics import root_mean_squared_error

from .optimization import w_sample_diag
from .matrix_operations import vec2ten3
from .model_functionality import predict_score

def rmse(y1, y2): 
    return root_mean_squared_error(y1, y2)

def pll(y_pred, var_pred, y_true):
    """ Computes the predictive log-likelihood. """
    log_likelihoods = -0.5 * jnp.log(2 * jnp.pi * var_pred) - ((y_true - y_pred) ** 2) / (2 * var_pred)
    return jnp.mean(log_likelihoods).item()

def nll(y_pred, var_pred, y_true):
    """ Computes the negative log-likelihood. """
    return -pll(y_pred, var_pred, y_true)

def norm_frob(x):
    return jnp.sqrt((x * x).sum())

def l2_gb_loss(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape):
    scores = predict_score(x, kd, vec2ten3(w_vec, *w_shape), fmap)
    return 0.5*(beta_e*jnp.sum((y - scores)**2) + gamma_w*(w_vec * w_vec).sum())

def l2_loss_kl(w_mean, w_std, w_shape, kd, x, y, fmap, gamma_w, beta_e, key, n_samples):
    """ L2 loss with closed form KL divergence. """
    w_var = w_std**2
    kl_term = gamma_w * (w_var.sum() + (w_mean**2).sum()) - jnp.sum(jnp.log(w_var))
    keys = jax.random.split(key, n_samples)

    def sample_loss(subkey):
        w_sample = w_sample_diag(w_mean, w_std, subkey)
        scores = predict_score(x, kd, vec2ten3(w_sample, *w_shape), fmap)
        return beta_e * jnp.sum((y - scores)**2)
    
    likelihood_term = jax.vmap(sample_loss)(keys).sum()
    return 0.5 * (likelihood_term + kl_term)

def ecp(y_true, y_dist, alpha=0.9) -> float:
    """
    Empirical Coverage Probability at level alpha.
    
    y_true : (N,) or (N,1)
    y_dist : (N, S) predictive samples
    alpha : float, must be in (0, 1)
    """
    y_true = jnp.squeeze(y_true)
    lower = jnp.quantile(y_dist, (1 - alpha) / 2.0, axis=1)
    upper = jnp.quantile(y_dist, 1 - (1 - alpha) / 2.0, axis=1)
    return jnp.mean((y_true >= lower) & (y_true <= upper))

def wcpi(y_dist, alpha=0.9) -> float:
    """
    Mean width of the prediction interval at level alpha.
    
    y_dist : (N, S) predictive samples
    alpha : float, must be in (0, 1)
    """
    lower = jnp.quantile(y_dist, (1 - alpha) / 2.0, axis=1)
    upper = jnp.quantile(y_dist, 1 - (1 - alpha) / 2.0, axis=1)
    return jnp.mean(upper - lower)

def rce(y_true, y_dist, alphas=None) -> float:
    """
    Regression Calibration Error (RCE).

    y_true : (N,) or (N,1)
    y_dist : (N, S) predictive samples
    alphas : Array, values must be in (0, 1)
    """
    if alphas is None:
        alphas = jnp.linspace(0.05, 0.95, 19)
    
    def ecp_alpha(alpha):
        return ecp(y_true, y_dist, alpha)

    coverages = jax.vmap(ecp_alpha)(alphas)
    return jnp.mean(jnp.abs(coverages - alphas))
