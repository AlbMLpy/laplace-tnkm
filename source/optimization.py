from itertools import product

import numpy as np
import jax.numpy as jnp
from jax import jit, hessian

from .evaluation import l2_gb_loss
from .matrix_operations import ten3tovec

def gd_update(w, dw, lr):
    w -= lr * dw
    return w

def adam_update(w_vec, dw, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1  # Time step update
    new_m = beta1*m + (1 - beta1)*dw
    new_v = beta2*v + (1 - beta2)*(dw**2)
    # Bias correction
    m_hat = new_m / (1 - beta1**t) 
    v_hat = new_v / (1 - beta2**t)
    # Parameter update
    new_params = w_vec - lr*m_hat/(jnp.sqrt(v_hat) + eps)
    return new_params, new_m, new_v, t

def std_transform(p):
    return jnp.log(jnp.exp(p) + 1)

def sample_w(w_mean, w_std):
    e_noise = jnp.array(np.random.randn(*w_mean.shape))
    return w_mean + e_noise*w_std

def hess_full_jax(w_ten, kd, x, y, fmap, gamma_w, beta_e, w_shape):
    hess_f = jit(hessian(l2_gb_loss, argnums=0), static_argnums=(4, 7))
    w_vec = ten3tovec(w_ten)
    hw = hess_f(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape)
    return hw

def full_grid(params):
    """
    Full grid search on hyper parameters.
    The function produces hyper parameters grid and names.
    
    Parameters
    ----------
    params : dict
        Dictionary of parameters names and variable values,
        e.g. {"A": [1, 2, 3], "B": [3, 4, 5]}.
    
    Returns
    -------
    grid, param_names : tuple
        Tuple-like object, with the following attributes.
    grid : set
        Set of configurations.
    param_names : tuple
        Tuple of all the parameters names.
    """
    param_names, param_values = zip(*params.items())
    grid = set(product(*param_values))
    return grid, param_names 
