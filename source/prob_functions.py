from typing import Optional

import numpy as np 
import jax.numpy as jnp

from .model_functionality import predict_score
from .matrix_operations import ten3tovec, vec2ten3

def w_sample(w_mean_vec, w_cholesky, seed=None):
    z = jnp.array(np.random.RandomState(seed).randn(w_cholesky.shape[1]))
    return w_mean_vec + w_cholesky.dot(z)

def predict_std(w_ten, w_cholesky, kd, x, fmap, beta_e, n_samples=10):
    w_mean_vec, w_shape = ten3tovec(w_ten), w_ten.shape
    preds = []
    for _ in range(n_samples):
        w_vec_sample = w_sample(w_mean_vec, w_cholesky)
        scores = predict_score(x, kd, vec2ten3(w_vec_sample, *w_shape), fmap)
        preds.append(scores[:, None])
    pred_std = jnp.std(jnp.hstack(preds), axis=1)
    std_err = 1 / jnp.sqrt(beta_e)
    return pred_std + std_err

def init_beta_e(beta_e: Optional[float] = None):
    beta_e_upd, cn, dn = beta_e is None, None, None
    if beta_e_upd:
        cn, dn = 1, 1
        beta_e = cn / dn
    return beta_e, beta_e_upd, cn, dn

def update_beta_e(c, d, w_cholesky, w_ten, kd, fmap, x, y, n_samples=10):
    w_mean_vec, w_shape = ten3tovec(w_ten), w_ten.shape
    mean_train_err = 0.0
    for _ in range(n_samples):
        w_vec_sample = w_sample(w_mean_vec, w_cholesky)
        scores = predict_score(x, kd, vec2ten3(w_vec_sample, *w_shape), fmap) 
        mean_train_err += jnp.sum((y - scores)**2)
    mean_train_err /= n_samples
    c += 0.5 * x.shape[0]
    d += 0.5 * mean_train_err
    return c, d, c / max(d, 1e-8)

def init_gamma_w(gamma_w: Optional[float] = None):
    gamma_w_upd, an, bn = gamma_w is None, None, None
    if gamma_w_upd:
        an, bn = 1, 1
        gamma_w = an / bn
    return gamma_w, gamma_w_upd, an, bn

def update_gamma_w(a, b, w_cov, w_vec):
    w_cov_diag = w_cov if w_cov.ndim == 1 else w_cov.diagonal()
    a += 0.5 * w_vec.size
    b += 0.5 * ((w_vec * w_vec).sum() + w_cov_diag.sum())
    return a, b, a / max(b, 1e-8)
