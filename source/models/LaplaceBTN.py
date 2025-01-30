from typing import Optional

import numpy as np
import jax.numpy as jnp
from jax import jit, grad, jacrev
from sklearn.utils.validation import check_X_y

from ..cpr import cpr
from ..optimization import gd_update
from ..features import Feature, PPFeature
from ..models.AbstractBTN import AbstractBTN
from ..model_functionality import predict_score, init_weights

class LaplaceBTN(AbstractBTN):
    def __init__(
        self, 
        rank: int = 1, 
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        std_err: float = 0.01,
        std_w: float = 1,
        seed: Optional[int] = None,
        opt_params: dict = {'train_mode': 'ALS'}
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, 
            std_err, std_w, seed, opt_params,
        )
        
        self.init_type = None
        self.alpha = (self.std_err / self.std_w)**2
        self.ww_reg = False
        self.pd_nsamples = 10 ######################## Need to change later on! ########################
    
    def _cov(self, other_params):
        hess_f = jit(jacrev(jacrev(l2_reg_loss, argnums=0), argnums=0), static_argnums=(4,))
        hw = hess_f(self.w_mean, *other_params).reshape((np.prod(self.w_shape),)*2)
        hw = jnp.linalg.inv(hw)
        L = jnp.linalg.cholesky(hw) ############ Not Sure about this! ############
        return hw, L

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        self._fmap = self._prepare_fmap()
        X, y = jnp.array(X), jnp.array(y)
        self.w_shape = (X.shape[-1], self.m_order, self.rank)
        # MAP estimation:
        if self.opt_params['train_mode'] == 'ALS':
            pinv = X.shape[-1] == 1
            self.w_mean, self.kd = cpr(
                X, y, self._quantized, self.m_order, self._fmap, 
                self.rank, self.init_type, self.n_epoch, self.alpha, self.seed, 
                self._dtype, xy_test, None, pinv=pinv, ww_reg=self.ww_reg,
            )
            other_params = (self.kd, X, y, self._fmap, self.alpha)
        elif self.opt_params['train_mode'] == 'GD':
            grad_w = jit(grad(l2_reg_loss, argnums=0), static_argnums=(4,))
            weights, self.kd = init_weights(
                self.m_order, 
                self.rank, 
                X.shape[-1], 
                q_base=2 if self._quantized else None, 
                seed=self.seed, 
                init_type=self.init_type
            )
            other_params = (self.kd, X, y, self._fmap, self.alpha)
            # Training loop:
            loss_list = []
            loss_list.append(l2_reg_loss(weights, *other_params))
            for _ in range(self.n_epoch):
                dw = grad_w(weights, *other_params)
                weights = gd_update(weights, dw, self.opt_params['lr'])
                loss_list.append(l2_reg_loss(weights, *other_params))
            self._loss_list = loss_list
            self.w_mean = weights
        else:
            raise ValueError(f'Bad train_mode = "{self.opt_params["train_mode"]}". See docs.')
        # Covariance matrix estimation:
        self.w_cov, self.L = self._cov(other_params)
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
                w_sample = w_mean_vec + self.L.dot(e_normal)
                prediction = predict_score(X, self.kd, w_sample.reshape(self.w_shape), self._fmap)
                preds.append(prediction[:, None])
            pred_std = self.std_err + jnp.std(jnp.hstack(preds), axis=1)
        elif std_mode == 'linearize':
            g_jac = jit(jacrev(predict_score, argnums=2), static_argnums=(3,))
            g = g_jac(X, self.kd, self.w_mean, self._fmap).reshape(X.shape[0], -1)
            pred_cov = self.std_err**2 + g.dot(self.w_cov.dot(g.T))
            pred_std = jnp.sqrt(pred_cov.diagonal())
        else:
            raise ValueError(f'Bad std_sampling = "{std_mode}". See docs.')
        return pred_std
    
def l2_reg_loss(weights, kd, x, y, fmap, alp):
    """x: (N, d), y: (N,), w: (d, I, R) """
    scores = predict_score(x, kd, weights, fmap) # (N,)
    loss = 0.5*(jnp.sum((y - scores)**2) + alp * (weights * weights).sum())
    return loss
