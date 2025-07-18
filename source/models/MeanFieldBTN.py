from typing import Optional

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
from sklearn.utils.validation import check_X_y

from .AbstractBTN import AbstractBTN
from ..features import Feature, PPFeature
from ..model_functionality import predict_score
from ..matrix_operations import vec2ten3
from ..optimization import gd_update, std_transform, w_sample_diag

class MeanFieldBTN(AbstractBTN):
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
        kl_closed_form: bool = False,
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, beta_e, gamma_w, seed, 
            opt_params, n_epoch_vi, pd_samples, beta_e_samples, tracker,
        )
        self.p_ids = ['m', 'p'] 
        self.n_loss_samples = n_loss_samples
        self._loss = l2_reg_loss_mf_closed_form_kl if kl_closed_form else l2_reg_loss_mf 
        self._loss_key = jax.random.PRNGKey(np.random.RandomState(seed).randint(1e18))

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        X, y = jnp.array(X), jnp.array(y)
        # Initialize gradient functions and parameters:
        self.w_shape = (X.shape[-1], self.m_order, self.rank)
        grad_f, self.params = self._init_fit()
        self.loss_list = []
        # VI training loop:
        for _ in range(self.n_epoch_vi):
            self._loss_key, subkey = jax.random.split(self._loss_key)
            self._update_w(X, y, grad_f, subkey)
            if not self.upd_beta_e and not self.upd_gamma_w:
                break
            if self.upd_gamma_w:
                self._update_gamma_w()
            if self.upd_beta_e:
                self._update_beta_e(X, y)
        self.is_fitted_ = True
        return self

    def _init_fit(self):
        key = jax.random.PRNGKey(np.random.RandomState(self.seed).randint(1e18))
        grad_f, params, loss_f = {}, {}, self._loss
        for i, p_id in enumerate(self.p_ids):
            grad_f[p_id] = jit(grad(loss_f, argnums=i), static_argnums=(2, 6, 9))
            key, subkey = jax.random.split(key)
            if p_id == 'm': 
                params[p_id] = jax.random.normal(subkey, (np.prod(self.w_shape),))
            elif p_id == 'p': 
                params[p_id] = jax.random.normal(subkey, (np.prod(self.w_shape),))
        return grad_f, params
    
    def _postprocess(self):
        self.w_mean = vec2ten3(self.params['m'], *self.w_shape)
        self.w_cholesky = jnp.diag(std_transform(self.params['p'])) # Not efficient!

    def _update_w(self, X, y, grad_f, key, xy_test: Optional[tuple] = None):
        if xy_test: raise NotImplementedError()
        
        other_params = (self.w_shape, self.kd, X, y, self._fmap, 
            self.gamma_w, self.beta_e, self.n_loss_samples
        )
        self.loss_list.append(
            self._loss(
                *[self.params[i] for i in self.p_ids], *other_params, key=key,
            )
        )
        for _ in range(self.n_epoch):
            key, subkey = jax.random.split(key)
            for k in self.p_ids:
                dw = grad_f[k](
                    *[self.params[i] for i in self.p_ids], *other_params, key=subkey
                )
                self.params[k] = gd_update(self.params[k], dw, self.opt_params['lr'])
            self.loss_list.append(
                self._loss(
                    *[self.params[i] for i in self.p_ids], *other_params, key=subkey
                )
            )
        self._postprocess()
    

def l2_reg_loss_mf(w_mean_vec, p_std_vec, w_shape, kd, x, y, 
    fmap, gamma_w, beta_e, n_loss_samples, key, eps=1e-8,
):
    w_std_vec = std_transform(p_std_vec)
    loss = 0.0
    for _ in range(n_loss_samples):
        key, subkey = jax.random.split(key)
        w_vec_sample = w_sample_diag(w_mean_vec, w_std_vec, subkey)
        scores = predict_score(x, kd, vec2ten3(w_vec_sample, *w_shape), fmap) 
        diff = w_vec_sample - w_mean_vec
        loss += beta_e * jnp.sum((y - scores)**2) 
        loss += gamma_w * (w_vec_sample * w_vec_sample).sum()
        loss -= jnp.log(w_std_vec**2).sum() + diff.dot(diff / (w_std_vec**2 + eps))
    return 0.5 * loss

def l2_reg_loss_mf_closed_form_kl(w_mean_vec, p_std_vec, w_shape, kd, x, y, 
    fmap, gamma_w, beta_e, n_loss_samples, key,
):
    w_std_vec = std_transform(p_std_vec)
    w_var = w_std_vec**2
    loss = gamma_w * (w_var.sum() + (w_mean_vec*w_mean_vec).sum()) - jnp.log(jnp.prod(w_var))
    for _ in range(n_loss_samples):
        key, subkey = jax.random.split(key)
        w_vec_sample = w_sample_diag(w_mean_vec, w_std_vec, subkey)
        scores = predict_score(x, kd, vec2ten3(w_vec_sample, *w_shape), fmap) 
        loss += beta_e * jnp.sum((y - scores)**2)
    return 0.5 * loss
