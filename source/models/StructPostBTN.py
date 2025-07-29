from typing import Optional
from itertools import product

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, grad

from .MeanFieldBTN import MeanFieldBTN
from ..features import Feature, PPFeature
from ..model_functionality import predict_score
from ..matrix_operations import cpd_transform_vec, vec2ten3
from ..optimization import std_transform, w_sample_diag

class StructPostBTN(MeanFieldBTN):
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
        
        self.mode_names = ['d', 'I', 'R'] # data_dim, local feature dim, CPD rank;
        self.w_types = ['m', 'p'] # Mean, std related parameters;
        self.p_ids = [(wt, fn) for wt, fn in product(self.w_types, self.mode_names)]
        self._loss = l2_reg_loss_sp_closed_form_kl 

    def _init_fit(self):
        key = jax.random.PRNGKey(np.random.RandomState(self.seed).randint(1e18))
        grad_f, params, loss_f = {}, {}, self._loss
        mode2num = {k: v for k, v in zip(self.mode_names, self.w_shape)}
        for i, p_id in enumerate(self.p_ids):
            w_type, mode_name = p_id
            grad_f[p_id] = jit(grad(loss_f, argnums=i), static_argnums=(6, 10, 13))
            pk_shape = (mode2num[mode_name], self.m_rank) if 'm' == w_type else (mode2num[mode_name],)
            key, subkey = jax.random.split(key)
            params[p_id] = jax.random.normal(subkey, pk_shape)
        return grad_f, params
    
    def _postprocess(self):
        w_mean = cpd_transform_vec(*[self.params[k] for k in self.p_ids if 'm' in k])
        w_std = std_transform_sp(*[self.params[k] for k in self.p_ids if 'p' in k])
        self.w_mean = vec2ten3(w_mean, *self.w_shape)
        self.w_cholesky = jnp.diag(w_std) # Not efficient!
    
def std_transform_sp(pd, pi, pr):
    pd, pi, pr = map(std_transform, (pd, pi, pr))
    return jnp.kron(pd, jnp.kron(pi, pr))

def l2_reg_loss_sp_closed_form_kl(md, mi, mr, pd, pi, pr,
    w_shape, kd, x, y, fmap, gamma_w, beta_e, n_loss_samples, key,
):
    w_mean_vec = cpd_transform_vec(md, mi, mr)
    w_std_vec = std_transform_sp(pd, pi, pr)
    w_var = w_std_vec**2
    loss = gamma_w * (w_var.sum() + (w_mean_vec*w_mean_vec).sum()) - jnp.log(jnp.prod(w_var))
    for _ in range(n_loss_samples):
        key, subkey = jax.random.split(key)
        w_vec_sample = w_sample_diag(w_mean_vec, w_std_vec, subkey)
        scores = predict_score(x, kd, vec2ten3(w_vec_sample, *w_shape), fmap)  
        loss += beta_e * jnp.sum((y - scores)**2) 
    return 0.5 * loss
