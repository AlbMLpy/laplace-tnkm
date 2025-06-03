from collections import defaultdict
from functools import partial

import numpy as np

import optax
import jax
import jax.numpy as jnp
from jax import jit, hessian
from jax import random
from jax.flatten_util import ravel_pytree
from .model_functionality import low_rank_cov_estimation, cov_estimation

from .general_functions import update_results_dict
from .evaluation import norm_frob

def norm_nn(nn_w):
    return sum([norm_frob(v).item() for v in nn_w.values()])

def n_params(nn_w):
    return sum([np.prod(v.shape) for v in nn_w.values()])

def init_params(key, input_dim, hidden_dim, output_dim):
    k1, k2 = random.split(key)
    params = {
        "W1": random.normal(k1, (input_dim, hidden_dim)) * 0.5,
        "b1": jnp.zeros((hidden_dim,)),
        "W2": random.normal(k2, (hidden_dim, output_dim)) * 0.5,
        "b2": jnp.zeros((output_dim,))
    }
    return params

def forward_nn(params, x):
    h = jax.nn.relu(x @ params["W1"] + params["b1"])
    return (h @ params["W2"] + params["b2"])[:, 0]

def loss_fn(params, x, y, beta_e, gamma_w):
    preds = forward_nn(params, x)
    fit_term = jnp.sum((preds - y)**2)
    l2_reg = jnp.sum(params["W1"]**2) + jnp.sum(params["W2"]**2)
    return 0.5*(beta_e*fit_term + gamma_w*l2_reg)

def train_map_nn(
    params, x, y, beta_e, gamma_w, 
    use_linesearch=False, opt_method='adam', lr=1e-3, n_epoch=1000
):
    if opt_method == 'adam':
        optimizer = optax.adam(lr) 
    elif opt_method == 'sgd':
        optimizer = optax.sgd(lr) 
    if use_linesearch:
        opt = optax.chain(optimizer, optax.scale_by_backtracking_linesearch(15)) #optax.scale_by_backtracking_linesearch(10)
    else:
        opt = optax.chain(optimizer)
    opt_state = opt.init(params)
    loss_fn_one = partial(loss_fn, x=x, y=y, beta_e=beta_e, gamma_w=gamma_w)
    @jit
    def step(params, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y, beta_e, gamma_w)
        if use_linesearch:
            updates, opt_state = opt.update(
                grads, opt_state, params, value=loss, grad=grads, value_fn=loss_fn_one)
        else:
            updates, opt_state = opt.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads
    
    tracker = defaultdict(list)
    train_loss, grads = jax.value_and_grad(loss_fn)(params, x, y, beta_e, gamma_w)
    update_results_dict(tracker, loss=train_loss.item(), grad_norm=norm_nn(grads))
    for epoch in range(n_epoch):
        params, opt_state, train_loss, grads = step(params, opt_state)
        update_results_dict(tracker, loss=train_loss.item(), grad_norm=norm_nn(grads))
    return params, tracker

def flat_loss_fn(nn_w_vec, unravel_fn, x, y, beta_e, gamma_w):
    nn_w = unravel_fn(nn_w_vec)
    return loss_fn(nn_w, x, y, beta_e, gamma_w)

def nn_full_hess(nn_w, x, y, beta_e, gamma_w):
    nn_w_vec, unravel_fn = ravel_pytree(nn_w)
    w_hess = hessian(flat_loss_fn)(nn_w_vec, unravel_fn, x, y, beta_e, gamma_w)
    return w_hess

def extract_block_diag(M, block_sizes):
    n = M.shape[0]
    assert M.shape[0] == M.shape[1], "Matrix must be square"
    assert sum(block_sizes) == n, "Block sizes must sum to matrix size"
    result = jnp.zeros_like(M)
    start = 0
    for size in block_sizes:
        end = start + size
        block = M[start:end, start:end]
        result = result.at[start:end, start:end].set(block)
        start = end
    return result

def nn_block_hess(nn_w, x, y, beta_e, gamma_w):
    w_hess = nn_full_hess(nn_w, x, y, beta_e, gamma_w)
    block_sizes = [np.prod(v.shape) for v in nn_w.values()]
    return extract_block_diag(w_hess, block_sizes)
    
def nn_diag_hess(nn_w, x, y, beta_e, gamma_w):
    return jnp.diag(jnp.diagonal(nn_full_hess(nn_w, x, y, beta_e, gamma_w)))

def hess_cov_estimation_nn(nn_w, x, y, gamma_w, beta_e, hess_type: str, low_rank: bool, low_rank_th: float = 1e-3):
    if hess_type == 'full':
        w_hess = nn_full_hess(nn_w, x, y, beta_e, gamma_w)
    elif hess_type == 'block':
        w_hess = nn_block_hess(nn_w, x, y, beta_e, gamma_w)
    elif hess_type == 'mf':
        w_hess = nn_diag_hess(nn_w, x, y, beta_e, gamma_w)
    else:
        raise ValueError(f'Bad hess_type: {hess_type}')

    if low_rank: cov_f = partial(low_rank_cov_estimation, threshold=low_rank_th)
    else: cov_f = cov_estimation
    
    try: 
        w_cov, w_cholesky = cov_f(w_hess)
        return w_hess, w_cov, w_cholesky
    except Exception as e: 
        return w_hess, None, None
