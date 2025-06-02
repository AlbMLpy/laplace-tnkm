from functools import partial

import jax.numpy as jnp
from jax import Array, jit, vmap
from jax.typing import ArrayLike

@jit
def khatri_rao_row(a: ArrayLike, b: ArrayLike) -> Array:
    return vmap(jnp.kron)(a, b)

@jit
def cpd_transform_vec(md, mi, mr):
    v = khatri_rao_row(mi.T, mr.T)
    return khatri_rao_row(md.T, v).T.sum(axis=1)

@jit
def block_diag(matrices) -> ArrayLike:
    """
    Creates a block diagonal matrix from a list of matrices in JAX.
    """
    rows = sum(mat.shape[0] for mat in matrices)
    cols = sum(mat.shape[1] for mat in matrices)
    result = jnp.zeros((rows, cols))
    r_offset, c_offset = 0, 0
    for mat in matrices:
        r, c = mat.shape
        result = result.at[r_offset:r_offset+r, c_offset:c_offset+c].set(mat)
        r_offset += r
        c_offset += c
    return result

@jit
def factors2vec(w):
    d_dim = w.shape[0]
    krp = w[0, :, :].T
    for k in range(1, d_dim):
        krp = khatri_rao_row(w[k, :, :].T, krp)
    return krp.T.sum(axis=1)

@jit
def ten3tovec(w_ten): # DIR
    return w_ten.transpose(1, 2, 0).reshape(-1, order='F')

@partial(jit, static_argnums=[1, 2, 3])
def vec2ten3(w_vec, d_dim, m_order, rank):
    return w_vec.reshape(m_order, rank, d_dim, order='F').transpose(2, 0, 1)
