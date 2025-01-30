from typing import Optional
from itertools import product

import numpy as np
import jax.numpy as jnp
from jax import jit, grad

from .MeanFieldBTN import MeanFieldBTN
from ..features import Feature, PPFeature
from ..model_functionality import predict_score
from ..matrix_operations import cpd_transform_vec
from ..optimization import std_transform, sample_w

class StructPostBTN(MeanFieldBTN):
    def __init__(
        self, 
        rank: int = 1, 
        m_rank: int = 1, 
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        std_err: float = 0.01,
        std_w: float = 0.01,
        n_loss_samples: int = 30, 
        seed: Optional[int] = None,
        opt_params: dict = {'train_mode': 'GD', 'lr': 1e-3}
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, 
            std_err, std_w, n_loss_samples, seed, opt_params,
        )
        self.m_rank = m_rank
        
        self.mode_names = ['d', 'I', 'R'] # data_dim, local feature dim, CPD rank;
        self.w_types = ['m', 'p'] # Mean parameters, std related parameters;
        self.p_ids = [(wt, fn) for wt, fn in product(self.w_types, self.mode_names)]
        self._loss = l2_reg_loss_sp_closed_form_kl

    def _init_fit(self):
        grad_f, params, loss_f = {}, {}, self._loss
        mode2num = {key: val for key, val in zip(self.mode_names, self.w_shape)}
        for i, p_id in enumerate(self.p_ids):
            w_type, mode_name = p_id
            grad_f[p_id] = jit(grad(loss_f, argnums=i), static_argnums=(6, 10, 13))
            pk_shape = (mode2num[mode_name], self.m_rank) if 'm' == w_type else (mode2num[mode_name],)
            params[p_id] = jnp.array(np.random.randn(*pk_shape))
        return grad_f, params
    
    def _postprocess(self):
        w_mean = cpd_transform_vec(*[self.params[k] for k in self.p_ids if 'm' in k])
        w_std = std_transform_sp(*[self.params[k] for k in self.p_ids if 'p' in k])
        self.w_mean = w_mean.reshape(self.w_shape) # As a 3-d tensor: d*I*R
        self.w_cov = w_std**2 # Save only the diagonal elements
    
def std_transform_sp(pd, pi, pr):
    pd, pi, pr = map(std_transform, (pd, pi, pr))
    return jnp.kron(pd, jnp.kron(pi, pr))

def l2_reg_loss_sp_closed_form_kl(md, mi, mr, pd, pi, pr,
    w_shape, kd, x, y, fmap, beta_w, beta_e, n_loss_samples,
):
    w_mean, w_std = cpd_transform_vec(md, mi, mr), std_transform_sp(pd, pi, pr)
    w_var = w_std**2
    loss = beta_w * (w_var.sum() + (w_mean*w_mean).sum()) - jnp.log(jnp.prod(w_var))
    for _ in range(n_loss_samples):
        weights = sample_w(w_mean, w_std).reshape(w_shape, order='F')
        scores = predict_score(x, kd, weights, fmap) 
        loss += beta_e * jnp.sum((y - scores)**2) 
    return 0.5 * loss
