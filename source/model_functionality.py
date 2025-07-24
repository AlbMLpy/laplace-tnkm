from typing import Optional
from functools import partial

import numpy as np
import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from .features import FeatureMap
from .matrix_operations import vec2ten3, ten3tovec, khatri_rao_row

DIV_EPS = 1e-14

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
    and optional normalization strategies. Use 'q_base' parameter 
    to generate quantized representation of weights.
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
def get_fw_hadamard_mtx(
    x: Array, 
    kd: int, 
    w_ten: Array, 
    fmap: FeatureMap,
) -> Array:
    """ 
    Calculate the Hadamard product between 
    matrix multiplications of features and corresponding CPD cores.
    """
    fw_hadamard = jnp.ones((x.shape[0], w_ten.shape[-1]), dtype=w_ten.dtype)
    for ind, wk in enumerate(w_ten):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        fw_hadamard *= fmap(x[:, k], q).dot(wk)
    return fw_hadamard

@partial(jit, static_argnums=(3,))
def predict_score(
    x: Array, 
    kd: int, 
    w_ten: Array, 
    fmap: FeatureMap,
) -> Array:
    """ 
    Generate prediction scores for CPD based model. 
    """
    score = jnp.ones((x.shape[0], w_ten.shape[-1]), dtype=w_ten.dtype)
    for ind, wk in enumerate(w_ten):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        score *= fmap(x[:, k], q).dot(wk)
    return jnp.real(jnp.sum(score, 1))

@partial(jit, static_argnums=(3,))
def predict_score_linear(
    x: Array, 
    kd: int, 
    w_ten: Array, 
    fmap: FeatureMap, 
    w_ten_opt: Array,
) -> Array:
    """ 
    Generate prediction scores for linearized CPD based model. 
    """
    w_jacob_manual = jacob_cpd(w_ten_opt, kd, x, fmap)
    pred = predict_score(x, kd, w_ten_opt, fmap)
    pred += w_jacob_manual.dot(ten3tovec(w_ten - w_ten_opt))
    return pred

@jit
def prepare_system(
    fk_mtx: Array, 
    fw_hadamard: Array,
    y: Array,
) -> tuple[Array, Array]:
    """
    Prepare local linear system of equations for one CPD core.
    """
    Fk = khatri_rao_row(fw_hadamard, fk_mtx) # Fortran Order
    return Fk.T.conj().dot(Fk), Fk.T.conj().dot(y)

@partial(jit, static_argnums=(2, 5,))
def update_weights(
    w_ten: Array, 
    kd: int, 
    w_shape: ArrayLike, 
    x: Array, 
    y: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float, 
    fw_hadamard: Array,
) -> tuple[Array, Array]:
    """ 
    Full update of all the model's CPD cores (one sweep/epoch).
    """
    d_dim, m_order, rank = w_shape
    for ind in range(d_dim):
        # Preprocess:
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        wk, fk_mtx = w_ten[ind], fmap(x[:, k], q)
        fw_hadamard /= (fk_mtx.dot(wk) + DIV_EPS)
        # Solve linear system:
        A, b = prepare_system(fk_mtx, fw_hadamard, y)
        A += gamma_w / beta_e * jnp.eye(m_order*rank)
        sol = jnp.linalg.solve(A, b)
        wk = jnp.array(sol.reshape(m_order, rank, order='F')) # Fortran Order
        w_ten = w_ten.at[ind].set(wk)
        # Postprocess:
        fw_hadamard *= fk_mtx.dot(wk)
    return w_ten, fw_hadamard

def als_cpd(
    w_vec: Array, 
    kd: int, 
    w_shape: ArrayLike, 
    x: Array, 
    y: Array, 
    fmap: FeatureMap, 
    n_epoch: int, 
    gamma_w: float, 
    beta_e: float, 
    tracker: Optional[object] = None,
) -> Array:
    """
    Compute 'optimal' CPD-based model weights tensor using 
    Alternating Least Squares (ALS) algorithm.
    """
    w_ten = vec2ten3(w_vec, *w_shape)
    fw_hadamard = get_fw_hadamard_mtx(x, kd, w_ten, fmap)
    if tracker: tracker.track(w_ten, kd, fmap)
    for _ in range(n_epoch):
        w_ten, fw_hadamard = update_weights(
            w_ten, kd, w_shape, x, y, fmap, gamma_w, beta_e, fw_hadamard,
        )
        if tracker: tracker.track(w_ten, kd, fmap)
    return ten3tovec(w_ten)

@partial(jit, static_argnums=(3,))
def jacob_cpd(
    w_ten: Array, 
    kd: int, 
    x: Array, 
    fmap: FeatureMap,
) -> Array:
    """
    Compute the Jacobian matrix of the CPD-based model prediction function.
    """
    d_dim, m_order, rank = w_ten.shape
    p_core = m_order*rank
    jacob_mtx = jnp.empty((x.shape[0], d_dim*p_core))
    fw_hadamard = get_fw_hadamard_mtx(x, kd, w_ten, fmap)
    for ind, wk in enumerate(w_ten):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        phi_k = fmap(x[:, k], q)
        phi_w = phi_k.dot(wk)
        fw_hadamard /= (phi_w + DIV_EPS)
        fk = khatri_rao_row(fw_hadamard, phi_k)
        fw_hadamard *= phi_w
        jacob_mtx = jacob_mtx.at[:, ind*p_core: ind*p_core + p_core].set(fk)
    return jacob_mtx

def hess_cov_estimation(
    w_ten: Array, 
    kd: int, 
    x: Array, 
    y: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float, 
    hess_type: str, 
    hess_th: float = 1e-3,
) -> tuple[Array, Optional[Array], Optional[Array]]:
    """
    Compute the Hessian approximation of the L2 loss in the CPD-based model setting.
    """
    if hess_type == 'full':
        w_hess = hess_full(w_ten, kd, x, y, fmap, gamma_w, beta_e, mode='full')
    elif hess_type == 'gauss_newton':
        w_hess = hess_ggn(w_ten, kd, x, fmap, gamma_w, beta_e)
    elif hess_type == 'block':
        w_hess = hess_block_diag(w_ten, kd, x, fmap, gamma_w, beta_e)
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
        w_cov, w_cholesky = cov_f(w_hess)
        success = True
    except Exception as e: 
        success = False

    if success: 
        return w_hess, w_cov, w_cholesky
    else:
        return w_hess, None, None

@partial(jit, static_argnums=(3,))
def hess_ggn(
    w_ten: Array, 
    kd: int, 
    x: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float,
) -> Array:
    """ 
    Generalized Gauss-Newton (GGN) estimation of the Hessian.
    """
    jf = jacob_cpd(w_ten, kd, x, fmap) # N x DIR Jacobian matrix
    return beta_e*jf.T.conj().dot(jf) + gamma_w*jnp.eye(*(jf.shape[1],)*2)

@partial(jit, static_argnums=(3,)) 
def hess_block_diag(
    w_ten: Array, 
    kd: int, 
    x: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float,
) -> Array:
    """
    Block-Diagonal estimation of the Hessian. Independent CPD cores.
    """
    d_dim, m_order, rank = w_ten.shape
    p_core = m_order*rank
    res = jnp.zeros((d_dim*p_core, d_dim*p_core))
    fw_hadamard = get_fw_hadamard_mtx(x, kd, w_ten, fmap) 
    for ind, wk in enumerate(w_ten):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        phi_k = fmap(x[:, k], q)
        phi_w = phi_k.dot(wk)
        fw_hadamard /= (phi_w + DIV_EPS)
        fk = khatri_rao_row(fw_hadamard, phi_k)
        fw_hadamard *= phi_w
        off = ind*p_core
        res = res.at[off:off + p_core, off:off + p_core].set(
            beta_e*fk.T.conj().dot(fk) + gamma_w*jnp.eye(p_core, p_core))
    return res

@partial(jit, static_argnums=(3,))
def hess_diag(
    w_ten: Array, 
    kd: int, 
    x: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float,
) -> Array: 
    """
    Diagonal estimation of the Hessian. Related to Mean-Field Approximation.
    """
    w_hess_gn = hess_ggn(w_ten, kd, x, fmap, gamma_w, beta_e) # Naive
    return jnp.diag(jnp.diagonal(w_hess_gn))

@partial(jit, static_argnums=(3,))
def hess_last(
    w_ten: Array, 
    kd: int, 
    x: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float,
) -> Array:
    """
    Last block estimation of the Hessian.
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

def hess_full(
    w_ten: Array, 
    kd: int, 
    x: Array, 
    y: Array, 
    fmap: FeatureMap, 
    gamma_w: float, 
    beta_e: float, 
    mode: str = 'full',
) -> Array:
    """
    Full Hessian estimation.
    """
    d_dim, m_order, rank = w_ten.shape
    P = m_order*rank
    hess_w = jnp.zeros((d_dim*P, d_dim*P))
    T = get_fw_hadamard_mtx(x, kd, w_ten, fmap) 
    for k in range(d_dim):
        zk, qk = divmod(k, kd) # q starts from zero -> for fmap
        wk, fk_mtx = w_ten[zk], fmap(x[:, zk], qk)
        Tk = _get_fw_part_mtx(T, [fk_mtx.dot(wk),])
        for m in range(k, d_dim):
            if k == m: 
                Fk = khatri_rao_row(Tk, fk_mtx)
                hess_w = hess_w.at[k*P:k*P + P, k*P:k*P + P].set(
                    beta_e*Fk.T.conj().dot(Fk) + gamma_w*jnp.eye(P, P))
            else:
                if mode == 'block_diag': continue
                zm, qm = divmod(m, kd) # q starts from zero -> for fmap
                wm, fm_mtx = w_ten[zm], fmap(x[:, zm], qm)
                fwm = fm_mtx.dot(wm)
                Tm = _get_fw_part_mtx(T, [fwm,])
                Dkm = khatri_rao_row(Tm, Tk)
                Ekm = (T.sum(axis=1) - y)[:, None] * _get_fw_part_mtx(Tk, [fwm,])
                for r in range(rank):
                    for p in range(rank):
                        Jkmrp = Dkm[:, r + p*rank]
                        if r == p: 
                            Jkmrp += Ekm[:, r]
                        Hkmrp = fk_mtx.T.dot(Jkmrp[:, None] * fm_mtx)
                        rx, cx = k*P + r*m_order, m*P + p*m_order
                        hess_w = hess_w.at[rx:rx + m_order, cx:cx + m_order].set(beta_e*Hkmrp)
                        hess_w = hess_w.at[cx:cx + m_order, rx:rx + m_order].set(beta_e*Hkmrp.T)
    return hess_w 

def _get_fw_part_mtx(fw_hadamard: Array, fw_list: list[Array]) -> Array:
    fwh = fw_hadamard.copy()
    for fw in fw_list:
        fwh /= fw
    return fwh

def cov_estimation(w_hess: Array) -> tuple[Array, Array]:
    """
    Compute corresponding covariance matrix and 
    Cholesky factor based on a provided Hessian matrix.
    """
    w_cholesky = np.linalg.cholesky(w_hess)
    w_cholesky = np.linalg.pinv(w_cholesky.T)
    w_cov = w_cholesky.dot(w_cholesky.T)
    return w_cov, w_cholesky

def low_rank_cov_estimation(w_hess: Array, threshold: float = 1e-3) -> Array:
    """
    Compute corresponding low-rank covariance matrix and 
    Cholesky factor based on a provided Hessian matrix.
    """
    s, u = np.linalg.eigh(w_hess)
    mask = s >= threshold
    w_cholesky = u[:, mask] * (1/jnp.sqrt(s[mask]))[None, :]
    w_cov = w_cholesky.dot(w_cholesky.T)
    return w_cov, w_cholesky

def check_zero_cols(w_ten: Array) -> Array:
    """
    Compute mask showing which columns of the CPD cores are close to zero.
    This mask can be used for rank truncation.
    """
    d_dim, _, rank = w_ten.shape
    mask = jnp.empty((d_dim, rank), dtype=jnp.bool)
    for i, wk in enumerate(w_ten):
        mask = mask.at[i, :].set((jnp.abs(wk) < 1e-14).all(axis=0))
    return mask

def process_weights(w_ten: Array) -> tuple[Array, ArrayLike]:
    """
    Compute updated CPD weights tensor with truncated rank (if needed).
    Rank truncation is based on close-to-zero columns in the matrices.
    """
    _mask = check_zero_cols(w_ten)
    mask = ~_mask.all(axis=0)
    w_ten = w_ten[:, :, mask]
    if w_ten.shape[-1] < 1: 
        raise ValueError(f'Zero Rank! W shape: {w_ten.shape}')
    return w_ten, w_ten.shape
