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
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        beta_e: Optional[float] = 1e2,
        beta_w: Optional[float] = 1e2,
        seed: Optional[int] = None,
        opt_params: dict = {'train_mode': 'GD', 'lr': 1e-3},
        n_epoch_global: int = 3, ### Not clear ###
        pred_dist_n_samples: int = 5, ### Not clear ###
        beta_e_n_samples: int = 3, ### Not clear ###
        n_loss_samples: int = 30, ### Not clear ###
        kl_closed_form: bool = True, ### Not clear ###
        m_rank: int = 1, 
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, 
            beta_e, beta_w, seed, opt_params,
            n_epoch_global, pred_dist_n_samples, 
            beta_e_n_samples, n_loss_samples, kl_closed_form
        )
        self.m_rank = m_rank
        
        self.mode_names = ['d', 'I', 'R'] # data_dim, local feature dim, CPD rank;
        self.w_types = ['m', 'p'] # Mean, std related parameters;
        self.p_ids = [(wt, fn) for wt, fn in product(self.w_types, self.mode_names)]
        if kl_closed_form:
            self._loss = l2_reg_loss_sp_closed_form_kl 
        else:
            raise NotImplementedError(f'kl_closed_form = {kl_closed_form}.')

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
