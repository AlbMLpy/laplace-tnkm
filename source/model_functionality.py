from functools import partial
from typing import Optional, Callable

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

import numpy as np

from .features import FeatureMap
from .matrix_operations import khatri_rao_row, block_diag_jax

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
    and optional normalization strategies. Use q_base parameter to generate 
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
def get_fw_hadamard_mtx(x: ArrayLike, kd: int, weights: ArrayLike, fmap: FeatureMap) -> Array:
    """ 
    Calculate the Hadamard product of matrix multiplication between features and CPD cores.

    Parameters
    ----------
    x : ArrayLike
        Input training data: (n_samples, d_dim).
    
    kd : int
        Degree in the equation: m_order = q_base^(kd).

    weights : ArrayLike
        An array containing the initialized weights in CPD format.

    fmap: FeatureMap
        Mapping from a data feature x_k to new features: f(x_k).

    Returns
    -------
    result : Array
        Array of shape: (n_samples, rank)
    """

    fw_hadamard = jnp.ones((x.shape[0], weights.shape[-1]), dtype=weights.dtype)
    for ind, wk in enumerate(weights):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        fw_hadamard *= fmap(x[:, k], q).dot(wk)
    return fw_hadamard

@jit
def get_ww_hadamard_mtx(weights: ArrayLike) -> Array:
    """ 
    Calculate the Hadamard product of matrix multiplication between corresponding CPD cores.

    Parameters
    ----------
    weights : ArrayLike
        An array containing the initialized weights in CPD format.

    dtype : jnp.dtype, default=jnp.float64
        The data type of the array elements.

    Returns
    -------
    result : Array
        Array of shape: (rank, rank)
    """

    ww_hadamard = jnp.ones((weights.shape[-1],)*2, dtype=weights.dtype)
    for wk in weights:
        ww_hadamard *= wk.T.conj().dot(wk)
    return ww_hadamard

@jit
def _prepare_system(
    fk_mtx: ArrayLike, 
    fw_hadamard: ArrayLike,
    y: ArrayLike,
) -> tuple[Array, Array]:
    Fk = khatri_rao_row(fw_hadamard, fk_mtx) # Fortran Ordering
    return Fk.T.conj().dot(Fk), Fk.T.conj().dot(y)

@partial(jit, static_argnums=(5, 6))
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
    A, b = _prepare_system(fk_mtx, fw_hadamard, y)
    if ww_reg:
        A += alpha * jnp.kron(ww_hadamard, jnp.eye(f_dim)) # Fortran Ordering
    else:
        A += alpha * jnp.eye(f_dim*rank)
    sol = jnp.linalg.pinv(A).dot(b) if pinv else jnp.linalg.solve(A, b)
    return sol.reshape(f_dim, rank, order='F') # Fortran Ordering

@partial(jit, static_argnums=(5, 8, 9))
def update_weights(
    x: ArrayLike, 
    y: ArrayLike,
    alpha: float,
    kd: int,
    weights: ArrayLike,
    fmap: FeatureMap,
    fw_hadamard: ArrayLike,
    ww_hadamard: ArrayLike,
    pinv: bool = False,
    ww_reg: bool = True,
) -> tuple[Array, Array, Array]:
    """ 
    Full update of model weights in CPD format.
    """
    for ind in range(weights.shape[0]):
        # Preprocess:
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        wk, fk_mtx = weights[ind], fmap(x[:, k], q)
        fw_hadamard /= fk_mtx.dot(wk) 
        ww_hadamard /= wk.T.conj().dot(wk) 
        # Solve linear system:
        wk = get_updated_als_factor(fk_mtx, fw_hadamard, ww_hadamard, y, alpha, pinv, ww_reg)
        weights = weights.at[ind].set(wk)
        # Postprocess:
        fw_hadamard *= fk_mtx.dot(wk)
        ww_hadamard *= wk.T.conj().dot(wk)
    return weights, fw_hadamard, ww_hadamard

@partial(jit, static_argnums=(4,))
def cov_block_diag(
    x: ArrayLike, 
    alpha: float,
    kd: int,
    weights: ArrayLike,
    fmap: FeatureMap,
):
    fw_hadamard = get_fw_hadamard_mtx(x, kd, weights, fmap) ### Not effective ###
    d, I, R = weights.shape
    blocks, blocks_l = [], []
    for ind in range(d):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        wk, fk_mtx = weights[ind], fmap(x[:, k], q)
        fw_hadamard /= fk_mtx.dot(wk) 
        Fk = khatri_rao_row(fw_hadamard, fk_mtx)
        block_inv = jnp.linalg.pinv(
            Fk.T.dot(Fk) 
            + alpha*jnp.eye(I*R, I*R)
        )
        blocks.append(block_inv)
        blocks_l.append(jnp.linalg.cholesky(block_inv))
        fw_hadamard *= fk_mtx.dot(wk)
    hw = block_diag_jax(blocks)
    L = block_diag_jax(blocks_l)
    return hw, L

@partial(jit, static_argnums=(3,))
def predict_score(
    x: ArrayLike, 
    kd: int, 
    weights: ArrayLike, 
    fmap: FeatureMap
) -> Array:
    """ 
    Generate prediction scores for CPD based model. 
    """
    n_samples, rank = x.shape[0], weights.shape[-1]
    score = jnp.ones((n_samples, rank), dtype=weights.dtype)
    for ind, wk in enumerate(weights):
        k, q = divmod(ind, kd) # q starts from zero -> for fmap
        score *= fmap(x[:, k], q).dot(wk)
    return jnp.real(jnp.sum(score, 1))

def run_callback(
    x: ArrayLike, 
    y: ArrayLike, 
    alpha: float, 
    kd: int, 
    weights: ArrayLike,  
    fmap: FeatureMap, 
    xy_test: Optional[tuple] = None,
    callback: Optional[Callable] = None,
) -> None:
    """
    Calculates user-defined callback function.
    """
    if callback:
        y_yp = None
        if xy_test:
            x_test, y_test = xy_test
            y_pred_test = predict_score(x_test, kd, weights, fmap)
            y_yp = y_test, y_pred_test
        y_pred = predict_score(x, kd, weights, fmap)
        callback(dict(y=y, y_pred=y_pred, weights=weights, alpha=alpha, y_yp=y_yp))
