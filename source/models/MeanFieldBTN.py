from typing import Optional
from itertools import product

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacrev
from sklearn.utils.validation import check_X_y

from .AbstractBTN import AbstractBTN
from ..features import Feature, PPFeature
from ..model_functionality import predict_score
from ..optimization import gd_update, std_transform, sample_w

# Looks like in paper the used non-closed form loss + 0, 1 init

class MeanFieldBTN(AbstractBTN):
    def __init__(
        self, 
        rank: int = 1, 
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
            std_err, std_w, seed, opt_params,
        )
        self.n_loss_samples = n_loss_samples
        self.kd = 1 ######################## Need to change later on! ########################
        self.pd_nsamples = 10 ######################## Need to change later on! ########################
        
        self.p_ids = ['m', 'p'] # Mean parameters, std related parameters;
        self._loss = l2_reg_loss_mf # l2_reg_loss_mf_closed_form_kl

    def _init_fit(self):
        grad_f, params, loss_f = {}, {}, self._loss
        for i, p_id in enumerate(self.p_ids):
            grad_f[p_id] = jit(grad(loss_f, argnums=i), static_argnums=(2, 6, 9))
            #params[p_id] = jnp.array(np.random.randn(np.prod(self.w_shape)))
            if p_id == 'm': params[p_id] = jnp.zeros(np.prod(self.w_shape))
            elif p_id == 'p': params[p_id] = jnp.ones(np.prod(self.w_shape))
        return grad_f, params
    
    def _postprocess(self):
        self.w_mean = self.params['m'].reshape(self.w_shape) # As a 3-d tensor: d*I*R
        self.w_cov = std_transform(self.params['p'])**2 # Save only the diagonal elements

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        self._fmap = self._prepare_fmap()
        X, y = jnp.array(X), jnp.array(y)
        loss_f = self._loss 
        self.w_shape = (X.shape[-1], self.m_order, self.rank)
        beta_w, beta_e = 1 / self.std_w**2, 1 / self.std_err**2
        other_params = (self.w_shape, self.kd, X, y, 
            self._fmap, beta_w, beta_e, self.n_loss_samples
        )
        # Initialize gradient functions and parameters:
        grad_f, params = self._init_fit()
        # Training loop:
        loss_list = []
        loss_list.append(loss_f(*[params[i] for i in self.p_ids], *other_params))
        for _ in range(self.n_epoch):
            for k in self.p_ids:
                dw = grad_f[k](*[params[i] for i in self.p_ids], *other_params)
                params[k] = gd_update(params[k], dw, self.opt_params['lr'])
            loss_list.append(loss_f(*[params[i] for i in self.p_ids], *other_params))
        self._loss_list = loss_list
        self.params = params
        # Postprocessing:
        self._postprocess()
        self.is_fitted_ = True
        return self
    
    def _predict_mean(self, X):
        return predict_score(X, self.kd, self.w_mean, self._fmap)

    def _predict_std(self, X, std_mode: Optional[str] = None):
        if std_mode == 'sampling':
            preds = []
            w_mean_vec = self.w_mean.reshape(-1)
            for _ in range(self.pd_nsamples):
                e_normal = jnp.array(np.random.randn(w_mean_vec.size)) ######################## Need to change later on! ########################
                w_sample = w_mean_vec + jnp.sqrt(self.w_cov) * e_normal
                prediction = predict_score(X, self.kd, w_sample.reshape(self.w_shape), self._fmap)
                preds.append(prediction[:, None])
            pred_std = self.std_err + jnp.std(jnp.hstack(preds), axis=1)
        elif std_mode == 'linearize':
            g_jac = jit(jacrev(predict_score, argnums=2), static_argnums=(3,))
            g = g_jac(X, self.kd, self.w_mean, self._fmap).reshape(X.shape[0], -1)
            pred_cov = self.std_err**2 + g.dot(self.w_cov[:, None] * g.T)
            pred_std = jnp.sqrt(pred_cov.diagonal())
        else:
            raise ValueError(f'Bad std_sampling = "{std_mode}". See docs.')
        return pred_std
    

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
