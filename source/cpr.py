from functools import partial
from typing import Optional, Callable

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

from .features import FeatureMap
from .model_functionality import (
    init_weights,
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    update_weights,
    run_callback,
)

Q_BASE = 2

#@partial(jit, static_argnums=(4, 10))
def cpr(
    x: ArrayLike, 
    y: ArrayLike,
    quantized: bool, 
    m_order: int,
    fmap: FeatureMap,
    rank: int,
    init_type: str,
    n_epoch: int,
    alpha: float,
    seed: Optional[int] = None,
    dtype: jnp.dtype = jnp.float64,
    xy_test: Optional[tuple] = None,
    callback: Optional[Callable] = None,
    pinv: bool = False,
    ww_reg: bool = True,
) -> tuple[Array, int]:
    q_base = Q_BASE if quantized else None
    weights, k_d = init_weights(m_order, rank, x.shape[-1], q_base, init_type, seed, dtype)
    fw_hadamard = get_fw_hadamard_mtx(x, k_d, weights, fmap)
    ww_hadamard = get_ww_hadamard_mtx(weights)
    run_callback(x, y, alpha, k_d, weights, fmap, xy_test, callback)
    for _ in range(n_epoch):
        weights, fw_hadamard, ww_hadamard = update_weights(x, y, alpha, k_d, weights, fmap, fw_hadamard, ww_hadamard, pinv, ww_reg)
        run_callback(x, y, alpha, k_d, weights, fmap, xy_test, callback)
    return weights, k_d
