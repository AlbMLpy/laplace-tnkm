from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from sklearn.utils.validation import check_X_y

from .AbstractBTN import AbstractBTN
from ..features import Feature, PPFeature
from ..model_functionality import predict_score
from ..optimization import gd_update, std_transform, sample_w

# Looks like in paper the used non-closed KL form loss + (0, 1 init)

class MeanFieldBTN(AbstractBTN):
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
        kl_closed_form: bool = False, ### Not clear ###
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, 
            beta_e, beta_w, seed, opt_params,
            n_epoch_global, pred_dist_n_samples, 
            beta_e_n_samples,
        )
        self.n_loss_samples = n_loss_samples
        self.p_ids = ['m', 'p'] # Mean, std related parameters;
        self._loss = l2_reg_loss_mf_closed_form_kl if kl_closed_form else l2_reg_loss_mf 

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        X, y = jnp.array(X), jnp.array(y)
        # Initialize gradient functions and parameters:
        self.w_shape = (X.shape[-1], self.m_order, self.rank)
        grad_f, self.params = self._init_fit()
        self.loss_list = []
        # Global training loop:
        for _ in range(self.n_epoch_global):
            self._update_w(X, y, grad_f)
            if not self.upd_beta_e and not self.upd_beta_w:
                break
            if self.upd_beta_w:
                self._update_beta_w()
            if self.upd_beta_e:
                self._update_beta_e(X, y)
        self.is_fitted_ = True
        return self
    
    def _w_sample(self, w_mean_vec):
        e_normal = jnp.array(np.random.randn(w_mean_vec.size)) # Need to change later on! #
        w_sample = w_mean_vec + jnp.sqrt(self.w_cov) * e_normal
        return w_sample
    
    def _linearized(self, g):
        return jnp.sqrt((self.w_cov[None, :] * g * g).sum(axis=1))

    def _init_fit(self):
        grad_f, params, loss_f = {}, {}, self._loss
        for i, p_id in enumerate(self.p_ids):
            grad_f[p_id] = jit(grad(loss_f, argnums=i), static_argnums=(2, 6, 9))
            if p_id == 'm': 
                params[p_id] = jnp.zeros(np.prod(self.w_shape)) # jnp.array(np.random.randn(np.prod(self.w_shape)))
            elif p_id == 'p': 
                params[p_id] = jnp.ones(np.prod(self.w_shape))
        return grad_f, params
    
    def _postprocess(self):
        self.w_mean = self.params['m'].reshape(self.w_shape) # As a 3-d tensor: d*I*R
        self.w_cov = std_transform(self.params['p'])**2 # Save only the diagonal elements

    def _update_w(self, X, y, grad_f):
        other_params = (self.w_shape, self.kd, X, y, self._fmap, 
            self.beta_w, self.beta_e, self.n_loss_samples
        )
        self.loss_list.append(self._loss(*[self.params[i] for i in self.p_ids], *other_params))
        for _ in range(self.n_epoch):
            for k in self.p_ids:
                dw = grad_f[k](*[self.params[i] for i in self.p_ids], *other_params)
                self.params[k] = gd_update(self.params[k], dw, self.opt_params['lr'])
            self.loss_list.append(self._loss(*[self.params[i] for i in self.p_ids], *other_params))
        self._postprocess()
    

def l2_reg_loss_mf(w_mean, p_std, w_shape, kd, x, y, 
    fmap, beta_w, beta_e, n_loss_samples, eps=1e-8,
):
    loss = 0.0
    for _ in range(n_loss_samples):
        w_std = std_transform(p_std)
        weights = sample_w(w_mean, w_std)
        diff = weights.reshape(-1, order='F') - w_mean
        weights = weights.reshape(w_shape, order='F')
        scores = predict_score(x, kd, weights, fmap) 
        loss += beta_e * jnp.sum((y - scores)**2) 
        loss += beta_w * (weights * weights).sum()
        loss -= jnp.log(w_std**2).sum() + diff.dot(diff / (w_std**2 + eps))
    return 0.5 * loss

def l2_reg_loss_mf_closed_form_kl(w_mean, p_std, w_shape, kd, x, y, 
    fmap, beta_w, beta_e, n_loss_samples, eps=1e-8
):
    w_std = std_transform(p_std)
    w_var = w_std**2
    loss = beta_w * (w_var.sum() + (w_mean*w_mean).sum()) - jnp.log(jnp.prod(w_var))
    for _ in range(n_loss_samples):
        weights = sample_w(w_mean, w_std).reshape(w_shape)
        scores = predict_score(x, kd, weights, fmap) 
        loss += beta_e * jnp.sum((y - scores)**2)
    return 0.5 * loss
