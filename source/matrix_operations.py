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
