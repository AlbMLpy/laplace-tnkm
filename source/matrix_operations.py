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
def block_diag_jax(matrices) -> ArrayLike:
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
