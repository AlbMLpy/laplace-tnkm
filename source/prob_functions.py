from typing import Optional

import numpy as np 
import jax.numpy as jnp
from jax.typing import ArrayLike

from .features import FeatureMap
from .matrix_operations import ten3tovec, vec2ten3
from .model_functionality import predict_score, predict_score_linear

EPS_DIV = 1e-8

def init_beta_e(beta_e: Optional[float] = None):
    beta_e_upd, cn, dn = beta_e is None, None, None
    if beta_e_upd:
        cn, dn = 1, 1
        beta_e = cn / dn
    return beta_e, beta_e_upd, cn, dn

def init_gamma_w(gamma_w: Optional[float] = None):
    gamma_w_upd, an, bn = gamma_w is None, None, None
    if gamma_w_upd:
        an, bn = 1, 1
        gamma_w = an / bn
    return gamma_w, gamma_w_upd, an, bn

def update_gamma_w(a, b, w_cholesky, w):
    w_cov_diag = (w_cholesky * w_cholesky).sum(axis=1)
    a += 0.5 * w.size
    b += 0.5 * ((w * w).sum() + w_cov_diag.sum())
    return a, b, a / max(b, EPS_DIV)

def w_sample(w_mean_vec, w_cholesky, seed: Optional[int] = None):
    """
    Sampling from a Gaussian Posterior.
    Note: Mode depends on 'w_cholesky' matrix shape: last or all;
    """
    z = jnp.array(np.random.RandomState(seed).randn(w_cholesky.shape[1]))
    w_sample, n_par = w_mean_vec.copy(), w_cholesky.shape[0]
    w_sample = w_sample.at[-n_par:].set(w_sample[-n_par:] + w_cholesky.dot(z))
    return w_sample

def get_scores(
    w_mean_vec: ArrayLike, 
    w_vec_sample: ArrayLike, 
    w_shape: tuple, 
    kd: int, 
    x: ArrayLike, 
    fmap: FeatureMap, 
    pd_mode: str,
) -> ArrayLike:
    if pd_mode == 'lla':
        scores = predict_score_linear(
            x, kd, vec2ten3(w_vec_sample, *w_shape), fmap, 
            vec2ten3(w_mean_vec, *w_shape)
        )
    elif pd_mode == 'la':
        scores = predict_score(
            x, kd, vec2ten3(w_vec_sample, *w_shape), fmap
        )
    return scores

def predict_std(
    w_ten: ArrayLike, 
    w_cholesky: ArrayLike, 
    kd: int, 
    x: ArrayLike, 
    fmap: FeatureMap, 
    beta_e: Optional[float], 
    pd_mode: str, 
    n_samples: int,
    seed: Optional[int],
) -> ArrayLike:
    w_mean_vec, w_shape = ten3tovec(w_ten), w_ten.shape
    preds = []
    for _ in range(n_samples):
        w_vec_sample = w_sample(w_mean_vec, w_cholesky, seed)
        scores = get_scores(w_mean_vec, w_vec_sample, w_shape, kd, x, fmap, pd_mode)
        preds.append(scores[:, None])
    pred_std = jnp.std(jnp.hstack(preds), axis=1)
    if beta_e is None:
        return pred_std
    else:
        return pred_std + 1 / jnp.sqrt(beta_e)
    
def update_beta_e(
    c: float, 
    d: float, 
    w_ten: ArrayLike, 
    w_cholesky: ArrayLike, 
    kd: int, 
    x: ArrayLike, 
    y: ArrayLike, 
    fmap: FeatureMap, 
    pd_mode: str, 
    n_samples: int,
    seed: Optional[int],
):
    w_mean_vec, w_shape = ten3tovec(w_ten), w_ten.shape
    mean_train_err = 0.0
    for _ in range(n_samples):
        w_vec_sample = w_sample(w_mean_vec, w_cholesky, seed)
        scores = get_scores(w_mean_vec, w_vec_sample, w_shape, kd, x, fmap, pd_mode)
        mean_train_err += jnp.sum((y - scores)**2)
    mean_train_err /= n_samples
    c += 0.5 * x.shape[0]
    d += 0.5 * mean_train_err
    return c, d, c / max(d, EPS_DIV)
