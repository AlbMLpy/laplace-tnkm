from typing import Optional

import jax
import numpy as np
from flax import struct
import jax.numpy as jnp

from ..evaluation import l2_loss_kl
from .MeanFieldBTN import MeanFieldBTN
from ..optimization import std_transform
from ..features import Feature, PPFeature
from ..matrix_operations import cpd_transform_vec, vec2ten3

@struct.dataclass
class SPBTNParams:
    md: jnp.ndarray # mean vector - data_dim
    mi: jnp.ndarray # mean vector - local feature dim
    mr: jnp.ndarray # mean vector - CPD rank
    pd: jnp.ndarray # log-std vector - data_dim
    pi: jnp.ndarray # log-std vector - local feature dim
    pr: jnp.ndarray # log-std vector - CPD rank

class StructPostBTN(MeanFieldBTN):
    """
    Structured Posteriors Variational Inference for Bayesian Tensor Networks.
    """
    def __init__(
        self, 
        rank: int = 1, 
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        beta_e: Optional[float] = 1.0,
        gamma_w: Optional[float] = 1.0,
        seed: Optional[int] = None,
        opt_params: dict = {'train_mode': 'GD', 'lr': 1e-3},
        n_epoch_vi: int = 1,
        pd_samples: int = 30,
        beta_e_samples: int = 10,
        tracker: Optional[object] = None,
        n_loss_samples: int = 30, 
        m_rank: int = 1, 
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, beta_e, gamma_w, seed, 
            opt_params, n_epoch_vi, pd_samples, beta_e_samples, tracker,
            n_loss_samples,
        )
        self.m_rank = m_rank
        self._loss = l2_loss_kl_sp

    def _init_params(self):
        key = jax.random.PRNGKey(np.random.RandomState(self.seed).randint(1e18))
        keys = jax.random.split(key, num=6)
        shapes_m = [(mk, self.m_rank) for mk in self.w_shape]
        shapes_p = [(mk,) for mk in self.w_shape]
        shapes = shapes_m + shapes_p
        return SPBTNParams(
            *[0.5*jax.random.normal(key, shape) for key, shape in zip(keys, shapes)]
        )
    
    def _postprocess(self):
        w_mean = cpd_transform_vec(self.params.md, self.params.mi, self.params.mr)
        w_std = std_transform_sp(self.params.pd, self.params.pi, self.params.pr)
        self.w_mean = vec2ten3(w_mean, *self.w_shape)
        self.w_cholesky = jnp.diag(w_std) # Not efficient!
    
def std_transform_sp(pd, pi, pr):
    pd, pi, pr = map(std_transform, (pd, pi, pr))
    return jnp.kron(pd, jnp.kron(pi, pr))

def l2_loss_kl_sp(params, w_shape, kd, x, y, fmap, gamma_w, beta_e, key, n_samples):
    w_mean_vec = cpd_transform_vec(params.md, params.mi, params.mr)
    w_std_vec = std_transform_sp(params.pd, params.pi, params.pr)
    return l2_loss_kl(
        w_mean_vec, w_std_vec, w_shape, kd, x, y, 
        fmap, gamma_w, beta_e, key, n_samples,
    )
