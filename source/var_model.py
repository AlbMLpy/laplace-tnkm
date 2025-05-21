from functools import partial
from collections import defaultdict

import numpy as np 
import jax.numpy as jnp

from .model_functionality import check_zero_cols, get_fw_hadamard_mtx, hess_full
from .matrix_operations import khatri_rao_row, vec2ten3, ten3tovec
from .prob_functions import (
    update_gamma_w, 
    init_gamma_w,
    update_beta_e, 
    init_beta_e,
)
from .optimizer import als_cpd
from .evaluation import l2_gb_loss
from .general_functions import update_results_dict

def process_weights(w_vec, w_shape):
    w_ten = vec2ten3(w_vec, *w_shape)
    _mask = check_zero_cols(w_ten)
    mask = ~_mask.all(axis=0)
    w_ten = w_ten[:, :, mask]
    if w_ten.shape[-1] < 1: 
        raise ValueError(f'Zero Rank! W shape: {w_ten.shape}')
    return w_ten, w_ten.shape

def var_cpd_als(
    w_ten, kd, x, y, fmap, n_epoch, gamma_w, beta_e, 
    n_epoch_g, beta_e_samples, grad_w, hess_type, low_rank, low_rank_th
):
    beta_e, beta_e_upd, cn, dn = init_beta_e(beta_e)
    gamma_w, gamma_w_upd, an, bn = init_gamma_w(gamma_w)
    tracker = defaultdict(list)
    for _ in range(n_epoch_g):
        # MAP Estimation:
        w_vec, w_shape = ten3tovec(w_ten), w_ten.shape
        w_vec, _ = als_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w)
        w_ten, w_shape = process_weights(w_vec, w_shape)
        # Hessian/Covariance/Cholesky Evaluation:
        w_hess, w_cov, w_cholesky = hess_cov_estimation(
            w_ten, kd, x, y, fmap, gamma_w, beta_e, hess_type, low_rank, low_rank_th)
        # Weights Precision:
        if gamma_w_upd: an, bn, gamma_w = update_gamma_w(an, bn, w_cov, ten3tovec(w_ten))
        # Error Precision:
        if beta_e_upd: cn, dn, beta_e = update_beta_e(cn, dn, w_cholesky, w_ten, kd, fmap, x, y, beta_e_samples)
        # Track training:
        update_results_dict(
            tracker, loss=l2_gb_loss(ten3tovec(w_ten), kd, x, y, fmap, gamma_w, beta_e, w_shape).item(),
            gamma_w=gamma_w, beta_e=beta_e.item(),
        )
    return w_ten, kd, w_cov, w_cholesky, gamma_w, beta_e, tracker

def jacob_cpd(w_ten, kd, x, fmap):
    D, I, R = w_ten.shape
    P = I*R
    jacob_mtx = jnp.empty((x.shape[0], D*P))
    fw_hadamard = get_fw_hadamard_mtx(x, kd, w_ten, fmap)
    for ind, wk in enumerate(w_ten):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        phi_k = fmap(x[:, k], q)
        phi_w = phi_k.dot(wk)
        fw_hadamard /= (phi_w + 1e-14)
        fk = khatri_rao_row(fw_hadamard, phi_k)
        fw_hadamard *= phi_w
        jacob_mtx = jacob_mtx.at[:, ind*P: ind*P + P].set(fk)
    return jacob_mtx

def gn_hess_estimation(w_ten, kd, x, fmap, gamma_w, beta_e):
    w_jacob_manual = jacob_cpd(w_ten, kd, x, fmap)
    w_hess_gn = beta_e*w_jacob_manual.T.dot(w_jacob_manual) 
    w_hess_gn += gamma_w*jnp.eye(*w_hess_gn.shape)
    return w_hess_gn

def mf_hess_estimation(w_ten, kd, x, fmap, gamma_w, beta_e): # SUPER BAD VERSION!!! #
    w_hess_gn = gn_hess_estimation(w_ten, kd, x, fmap, gamma_w, beta_e)
    return jnp.diag(jnp.diagonal(w_hess_gn))

def cov_estimation(w_hess):
    w_cholesky = np.linalg.cholesky(w_hess)
    w_cholesky = np.linalg.pinv(w_cholesky.T)
    w_cov = w_cholesky.dot(w_cholesky.T)
    return w_cov, w_cholesky

def low_rank_cov_estimation(w_hess, threshold=1e-3):
    s, u = np.linalg.eigh(w_hess)
    mask = s >= threshold
    w_cholesky = u[:, mask] * (1/jnp.sqrt(s[mask]))[None, :]
    w_cov = w_cholesky.dot(w_cholesky.T)
    return w_cov, w_cholesky

def hess_cov_estimation(w_ten, kd, x, y, fmap, gamma_w, beta_e, hess_type: str, low_rank: bool, low_rank_th: float = 1e-3):
    if hess_type == 'full':
        w_hess = hess_full(w_ten, kd, x, y, fmap, gamma_w, beta_e, mode='full')
    elif hess_type == 'gauss_newton':
        w_hess = gn_hess_estimation(w_ten, kd, x, fmap, gamma_w, beta_e)
    elif hess_type == 'block':
        w_hess = hess_full(w_ten, kd, x, y, fmap, gamma_w, beta_e, mode='block_diag')
    elif hess_type == 'mf':
        w_hess = mf_hess_estimation(w_ten, kd, x, fmap, gamma_w, beta_e)
    else:
        raise ValueError(f'Bad hess_type: {hess_type}')
    
    if low_rank:
        cov_f = partial(low_rank_cov_estimation, threshold=low_rank_th)
    else:
        cov_f = cov_estimation
    
    try: 
        w_cov, w_cholesky = cov_f(w_hess); success = True;
    except Exception as e: 
        success = False

    if success: 
        return w_hess, w_cov, w_cholesky
    else:
        return w_hess, None, None
