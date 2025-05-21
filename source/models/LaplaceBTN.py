from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacrev
from sklearn.utils.validation import check_X_y

from ..cpr import cpr
from ..optimization import gd_update
from ..features import Feature, PPFeature
from ..models.AbstractBTN import AbstractBTN
from ..model_functionality import (
    init_weights,
    predict_score,
    cov_block_diag,
)

class LaplaceBTN(AbstractBTN):
    def __init__(
        self, 
        rank: int = 1, 
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        beta_e: Optional[float] = 1e2,
        beta_w: Optional[float] = 1e2,
        seed: Optional[int] = None,
        opt_params: dict = {'train_mode': 'ALS'},
        n_epoch_global: int = 3, ### Not clear ###
        pred_dist_n_samples: int = 5, ### Not clear ###
        beta_e_n_samples: int = 3, ### Not clear ###
        block_cov: bool = False, 
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, 
            beta_e, beta_w, seed, opt_params,
            n_epoch_global, pred_dist_n_samples, 
            beta_e_n_samples,
        )
        
        self.init_type = None
        self.alpha = self.beta_w / self.beta_e
        self.ww_reg = False
        self._loss = l2_reg_loss 
        self.block_cov = block_cov

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        X, y = jnp.array(X), jnp.array(y)
        self.w_shape = (X.shape[-1], self.m_order, self.rank)
        self.loss_list = []
        # Global training loop:
        for _ in range(self.n_epoch_global):
            self._update_w(X, y, xy_test)
            if not self.upd_beta_e and not self.upd_beta_w:
                break
            if self.upd_beta_w:
                self._update_beta_w()
            if self.upd_beta_e:
                self._update_beta_e(X, y)
            self.alpha = self.beta_w / self.beta_e
        self.is_fitted_ = True
        return self
    
    def _w_sample(self, w_mean_vec):
        e_normal = jnp.array(np.random.randn(w_mean_vec.size)) # Need to change later on! #
        w_sample = w_mean_vec + self.L.dot(e_normal)
        return w_sample
    
    def _linearized(self, g):
        return jnp.sqrt((g * g.dot(self.w_cov)).sum(axis=1))
    
    def _update_w(self, X, y, xy_test):
        if self.opt_params['train_mode'] == 'ALS':
            pinv = self.w_shape[0] == 1
            self.w_mean, self.kd = cpr(
                X, y, self._quantized, self.m_order, self._fmap, 
                self.rank, self.init_type, self.n_epoch, self.alpha, self.seed, 
                self._dtype, xy_test, None, pinv=pinv, ww_reg=self.ww_reg,
            )
            other_params = (self.kd, X, y, self._fmap, self.alpha)
        elif self.opt_params['train_mode'] == 'GD':
            grad_w = jit(grad(self._loss, argnums=0), static_argnums=(4,))
            weights, self.kd = init_weights(
                self.m_order, 
                self.rank, 
                self.w_shape[0], 
                q_base=2 if self._quantized else None, 
                seed=self.seed, 
                init_type=self.init_type
            )
            other_params = (self.kd, X, y, self._fmap, self.alpha)
            # Training loop:
            self.loss_list.append(self._loss(weights, *other_params))
            for _ in range(self.n_epoch):
                dw = grad_w(weights, *other_params)
                weights = gd_update(weights, dw, self.opt_params['lr'])
                self.loss_list.append(self._loss(weights, *other_params))
            self.w_mean = weights
        else:
            raise ValueError(f'Bad train_mode = "{self.opt_params["train_mode"]}". See docs.')
        # Covariance matrix estimation:
        if self.block_cov:
            self.w_cov, self.L = cov_block_diag(
                X, self.alpha, self.kd, self.w_mean, self._fmap
            )
        else:
            self.w_cov, self.L = self._cov_full(other_params)

    def nearest_positive_definite(self, matrix):
        eigval, eigvec = np.linalg.eigh(matrix)
        eigval[eigval < 0] = 0.1 # Replace negative eigenvalues with small positive values
        return eigvec @ np.diag(eigval) @ eigvec.T

    def _cov_full(self, other_params):
        hess_f = jit(jacrev(jacrev(self._loss, argnums=0), argnums=0), static_argnums=(4,))
        hw = hess_f(self.w_mean, *other_params).reshape((np.prod(self.w_shape),)*2)
        #print(self.w_mean)
        print(self.alpha)
        print(hw)
        hw = np.array(hw)
        hw = np.linalg.pinv(hw)
        #hw += np.eye(hw.shape[0]) * 1e-6
        #print(type(hw))
        hw = self.nearest_positive_definite(hw)
        L = np.linalg.cholesky(hw) # Not Sure about this! #
        return jnp.array(hw), jnp.array(L)
    
def l2_reg_loss(weights, kd, x, y, fmap, alp):
    """x: (N, d), y: (N,), w: (d, I, R) """
    scores = predict_score(x, kd, weights, fmap) # (N,)
    loss = 0.5*(jnp.sum((y - scores)**2) + alp * (weights * weights).sum())
    return loss
