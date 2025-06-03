from functools import partial
from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from .features import FeatureMap
from .general_functions import check_nan
from .matrix_operations import (
    vec2ten3,
    ten3tovec,
    khatri_rao_row,
)

def init_weights(
    m_order: int, 
    rank: int, 
    d_dim: int, 
    q_base: Optional[int] = None, 
    init_type: Optional[str] = None, 
    seed: Optional[int] = None,
    dtype: jnp.dtype = jnp.float64,
) -> tuple[Array, int]:
    """
    Initialize weights in CPD format using Normal Distribution 
    and optional normalizationcheck_zero_cols strategies. Use q_base parameter to generate 
    quantized representation of weights.

    Parameters
    ----------
    m_order : int
        The number of new generated features per 1 data feature. 
        In case of quantized version m_order must be a power of 2.

    rank : int
        The rank of the CP Decomposition based weights tensor.

    d_dim : int
        The number of features in the data.

    q_base : int, optional, default=None
        To use a quantized model set q_base=2. 
        To use non-quantized model set q_base=None.

    init_type : str, optional, default=None
        The normalization strategy for the weights:
            'k_mtx' - Normalize each matrix in the weights tensor.
            'kj_vec' - Normalize each vector in the matrices of the weights tensor.
            None - No normalization.

    seed : int, optional, default=None
        A seed for the random number generator to ensure reproducibility.

    dtype : jnp.dtype, default=jnp.float64
        The data type of the array elements.

    Returns
    -------
    weights : Array
        An array containing the initialized weights. Shape of array depends on q_base.

    kd : int
        Degree in the equation: m_order = q_base^(kd).

    Raises
    ------
    ValueError
        If the `init_type` provided is not recognized.
    """

    if (m_order & (m_order - 1)) and q_base:
        raise ValueError(f"m_order should be a power of 2, but it is {m_order}. ")
    random_state = np.random if seed is None else np.random.RandomState(seed)
    if q_base:
        kd = int(np.emath.logn(q_base, m_order)) # m_order = q_base^(kd) 
        weights = random_state.randn(d_dim*kd, q_base, rank)
    else:
        kd = 1
        weights = random_state.randn(d_dim, m_order, rank)
    weights = jnp.array(weights)
    naxis = weights.ndim - 2
    if init_type == 'k_mtx': # Matrix weights[k][:, :] is normalized
        weights /= jnp.linalg.norm(weights, ord=2, axis=(naxis, naxis + 1), keepdims=True)
    elif init_type == 'kj_vec': # Vector weights[k][:][j] is normalized
        weights /= jnp.linalg.norm(weights, ord=2, axis=naxis, keepdims=True)
    return weights.astype(dtype), kd

@partial(jit, static_argnums=(3,))
def get_fw_hadamard_mtx(x: ArrayLike, kd: int, w_ten: ArrayLike, fmap: FeatureMap) -> Array:
    """ 
    Calculate the Hadamard product of matrix multiplication between features and CPD cores.

    Parameters
    ----------
    x : ArrayLike
        Input training data: (n_samples, d_dim).
    
    kd : int
        Degree in the equation: m_order = q_base^(kd).

    w_ten : ArrayLike
        An array containing the initialized weights in CPD format.

    fmap: FeatureMap
        Mapping from a data feature x_k to new features: f(x_k).

    Returns
    -------
    result : Array
        Array of shape: (n_samples, rank)
    """

    fw_hadamard = jnp.ones((x.shape[0], w_ten.shape[-1]), dtype=w_ten.dtype)
    for ind, wk in enumerate(w_ten):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        fw_hadamard *= fmap(x[:, k], q).dot(wk)
    return fw_hadamard

@jit
def get_ww_hadamard_mtx(w_ten: ArrayLike) -> Array:
    """ 
    Calculate the Hadamard product of matrix multiplication between corresponding CPD cores.

    Parameters
    ----------
    w_ten : ArrayLike
        An array containing the initialized weights in CPD format.

    dtype : jnp.dtype, default=jnp.float64
        The data type of the array elements.

    Returns
    -------
    result : Array
        Array of shape: (rank, rank)
    """

    ww_hadamard = jnp.ones((w_ten.shape[-1],)*2, dtype=w_ten.dtype)
    for wk in w_ten:
        ww_hadamard *= wk.T.conj().dot(wk)
    return ww_hadamard

@jit
def prepare_system(
    fk_mtx: ArrayLike, 
    fw_hadamard: ArrayLike,
    y: ArrayLike,
) -> tuple[Array, Array]:
    Fk = khatri_rao_row(fw_hadamard, fk_mtx) # Fortran Ordering
    return Fk.T.conj().dot(Fk), Fk.T.conj().dot(y)

#@partial(jit, static_argnums=(5, 6))
def get_updated_als_factor(
    fk_mtx: ArrayLike, 
    fw_hadamard: ArrayLike,
    ww_hadamard: ArrayLike,
    y: ArrayLike,
    alpha: float,
    pinv: bool = False,
    ww_reg: bool = True,
) -> Array:
    """ 
    Solve custom linear system of equations.
    
    Parameters
    ----------
    fk_mtx : ArrayLike
        Feature matrix: (n_samples, mapping_dim).

    fw_hadamard : ArrayLike
        Helping Hadamard product matrix of Feature matrix times W_k  CPD core: (n_samples, rank).

    ww_hadamard : ArrayLike
        Helping Hadamard product matrix of W_k^T@W_k: (rank, rank).

    y : ArrayLike
        Target values array.

    alpha : float
        L2 regularization hyper-parameter.
        
    Returns
    -------
    result : Array
        Solution of LLS problem for 1 CPD core.
    """

    (_, f_dim), (rank, _) = fk_mtx.shape, ww_hadamard.shape
    A, b = prepare_system(fk_mtx, fw_hadamard, y)
    if ww_reg:
        A += alpha * jnp.kron(ww_hadamard, jnp.eye(f_dim)) # Fortran Ordering
    else:
        A += alpha * jnp.eye(f_dim*rank)
    sol = np.linalg.pinv(A).dot(b) if pinv else np.linalg.solve(A, b)
    return jnp.array(sol.reshape(f_dim, rank, order='F')) # Fortran Ordering

#@partial(jit, static_argnums=(5, 8, 9))
def update_weights(
    x: ArrayLike, 
    y: ArrayLike,
    gamma_w: float,
    beta_e: float,
    kd: int,
    w_ten: ArrayLike,
    fmap: FeatureMap,
    fw_hadamard: ArrayLike,
    ww_hadamard: ArrayLike,
    pinv: bool = False,
    ww_reg: bool = True,
) -> tuple[Array, Array, Array]:
    """ Full update of model weights in CPD format. """
    for ind in range(w_ten.shape[0]):
        # Preprocess:
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        wk, fk_mtx = w_ten[ind], fmap(x[:, k], q)
        fw_hadamard /= (fk_mtx.dot(wk) + 1e-14) # ZERO DIVISION?
        ww_hadamard /= (wk.T.conj().dot(wk) + 1e-14) # ZERO DIVISION?
        # Solve linear system:
        wk = get_updated_als_factor(fk_mtx, fw_hadamard, ww_hadamard, y, gamma_w / beta_e, pinv, ww_reg)
        w_ten = w_ten.at[ind].set(wk)
        # Postprocess:
        fw_hadamard *= fk_mtx.dot(wk)
        ww_hadamard *= wk.T.conj().dot(wk)
    return w_ten, fw_hadamard, ww_hadamard

@partial(jit, static_argnums=(3,))
def predict_score(
    x: ArrayLike, 
    kd: int, 
    w_ten: ArrayLike, 
    fmap: FeatureMap,
) -> Array:
    """ Generate prediction scores for CPD based model. """
    score = jnp.ones((x.shape[0], w_ten.shape[-1]), dtype=w_ten.dtype)
    for ind, wk in enumerate(w_ten):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        score *= fmap(x[:, k], q).dot(wk)
    return jnp.real(jnp.sum(score, 1))

@partial(jit, static_argnums=(3,))
def predict_score_linear(
    x: ArrayLike, 
    kd: int, 
    w_ten: ArrayLike, 
    fmap: FeatureMap, 
    w_ten_opt: ArrayLike,
):
    w_jacob_manual = jacob_cpd(w_ten_opt, kd, x, fmap)
    pred = predict_score(x, kd, w_ten_opt, fmap)
    pred += w_jacob_manual.dot(ten3tovec(w_ten - w_ten_opt))
    return pred

def als_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, tracker: Optional[object] = None):
    D, I, R = w_shape
    w_ten = vec2ten3(w_vec, D, I, R)
    fw_hadamard = get_fw_hadamard_mtx(x, kd, w_ten, fmap)
    if tracker: tracker.track()
    for ep in range(n_epoch):
        for ind in range(w_ten.shape[0]):
            # Preprocess:
            k, q = divmod(ind, kd) # q starts from zero -> for fmap
            wk, fk_mtx = w_ten[ind], fmap(x[:, k], q)
            fw_hadamard /= (fk_mtx.dot(wk) + 1e-14) # ZERO DIVISION?
            # Solve linear system:
            alpha = gamma_w / beta_e
            A, b = prepare_system(fk_mtx, fw_hadamard, y)
            A += alpha * jnp.eye(I*R)
            sol = jnp.linalg.solve(A, b)
            wk = jnp.array(sol.reshape(I, R, order='F')) # Fortran Ordering
            w_ten = w_ten.at[ind].set(wk)
            # Postprocess:
            fw_hadamard *= fk_mtx.dot(wk)
            check_nan(w_ten)
        if tracker: tracker.track()
    return ten3tovec(w_ten)

@partial(jit, static_argnums=(3,))
def jacob_cpd(
    w_ten: ArrayLike, 
    kd: int, 
    x: ArrayLike, 
    fmap: FeatureMap
):
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

def hess_cov_estimation(w_ten, kd, x, y, fmap, gamma_w, beta_e, hess_type: str, hess_th: float = 1e-3):
    if hess_type == 'full':
        w_hess = hess_full(w_ten, kd, x, y, fmap, gamma_w, beta_e, mode='full')
    elif hess_type == 'gauss_newton':
        w_hess = hess_ggn(w_ten, kd, x, fmap, gamma_w, beta_e)
    elif hess_type == 'block':
        w_hess = hess_full(w_ten, kd, x, y, fmap, gamma_w, beta_e, mode='block_diag')
    elif hess_type == 'mf':
        w_hess = hess_diag(w_ten, kd, x, fmap, gamma_w, beta_e)
    elif hess_type == 'last':
        w_hess = hess_last(w_ten, kd, x, fmap, gamma_w, beta_e)
    else:
        raise ValueError(f'Bad hess_type: {hess_type}')
    
    if hess_th:
        cov_f = partial(low_rank_cov_estimation, threshold=hess_th)
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
    
def hess_ggn(w_ten, kd, x, fmap, gamma_w: float, beta_e: float):
    """ 
    Generalized Gauss-Newton estimation of the Hessian.
    Note: Naive version! Do not use structure of any kind. 
    """
    w_jacob = jacob_cpd(w_ten, kd, x, fmap)
    w_hess_gn = beta_e*w_jacob.T.conj().dot(w_jacob)
    return w_hess_gn + gamma_w*jnp.eye(*w_hess_gn.shape)

def hess_diag(w_ten, kd, x, fmap, gamma_w: float, beta_e: float): 
    """
    Diagonal estimation of the Hessian. Related to Mean-Field Approximation.
    Note: Naive version! Do not use structure of any kind. 
    """
    w_hess_gn = hess_ggn(w_ten, kd, x, fmap, gamma_w, beta_e)
    return jnp.diag(jnp.diagonal(w_hess_gn))

def hess_last(w_ten, kd, x, fmap, gamma_w: float, beta_e: float):
    """
    Estimation of the last diagonal block of the Hessian. 
    Related to being Bayesian w.r.t. the last CPD core.
    """
    fw_hadamard = get_fw_hadamard_mtx(x, kd, w_ten, fmap)
    d_dim, m_order, rank = w_ten.shape
    last_ind = d_dim - 1
    k, q = divmod(last_ind, kd) # q starts from zero -> for fmap
    fk_mtx = fmap(x[:, k], q)
    fw_hadamard /= fk_mtx.dot(w_ten[last_ind]) 
    Fk = khatri_rao_row(fw_hadamard, fk_mtx)
    return beta_e*Fk.T.conj().dot(Fk) + gamma_w*jnp.eye(*(m_order*rank,)*2) 

def get_fw_part_mtx(fw_hadamard, fw_list):
    fwh = fw_hadamard.copy()
    for fw in fw_list:
        fwh /= fw
    return fwh

# Problems with jit!
def hess_full(
    w_ten, 
    kd, 
    x, 
    y, 
    fmap, 
    gamma_w: float, 
    beta_e: float, 
    mode: str = 'full'
):
    d, I, R = w_ten.shape
    P = I*R
    hess_w = jnp.zeros((d*P, d*P))
    T = get_fw_hadamard_mtx(x, kd, w_ten, fmap) 
    for k in range(d):
        zk, qk = divmod(k, kd) # q starts from zero -> for fmap
        wk, fk_mtx = w_ten[zk], fmap(x[:, zk], qk)
        Tk = get_fw_part_mtx(T, [fk_mtx.dot(wk),])
        for m in range(k, d):
            if k == m: 
                Fk = khatri_rao_row(Tk, fk_mtx)
                hess_w = hess_w.at[k*P:k*P + P, k*P:k*P + P].set(
                    beta_e*Fk.T.conj().dot(Fk) + gamma_w*jnp.eye(P, P))
            else:
                if mode == 'block_diag': continue
                zm, qm = divmod(m, kd) # q starts from zero -> for fmap
                wm, fm_mtx = w_ten[zm], fmap(x[:, zm], qm)
                fwm = fm_mtx.dot(wm)
                Tm = get_fw_part_mtx(T, [fwm,])
                Dkm = khatri_rao_row(Tm, Tk)
                Ekm = (T.sum(axis=1) - y)[:, None] * get_fw_part_mtx(Tk, [fwm,])
                for r in range(R):
                    for p in range(R):
                        Jkmrp = Dkm[:, r + p*R]
                        if r == p: 
                            Jkmrp += Ekm[:, r]
                        Hkmrp = fk_mtx.T.dot(Jkmrp[:, None] * fm_mtx)
                        rx, cx = k*P + r*I, m*P + p*I
                        hess_w = hess_w.at[rx:rx + I, cx:cx + I].set(beta_e*Hkmrp)
                        hess_w = hess_w.at[cx:cx + I, rx:rx + I].set(beta_e*Hkmrp.T)
    return hess_w 

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

def check_zero_cols(w_ten):
    d_dim, _, rank = w_ten.shape
    mask = jnp.empty((d_dim, rank), dtype=jnp.bool)
    for i, wk in enumerate(w_ten):
        mask = mask.at[i, :].set((jnp.abs(wk) < 1e-14).all(axis=0))
    return mask

def process_weights(w_ten: ArrayLike):
    _mask = check_zero_cols(w_ten)
    mask = ~_mask.all(axis=0)
    w_ten = w_ten[:, :, mask]
    if w_ten.shape[-1] < 1: 
        raise ValueError(f'Zero Rank! W shape: {w_ten.shape}')
    return w_ten, w_ten.shape
