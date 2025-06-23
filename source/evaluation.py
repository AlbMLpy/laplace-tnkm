import jax.numpy as jnp

from sklearn.metrics import root_mean_squared_error

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
