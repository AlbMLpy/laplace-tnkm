from typing import Optional

import jax.numpy as jnp
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..matrix_operations import ten3tovec, vec2ten3
from ..features import Feature, PPFeature, prepare_fmap
from ..model_functionality import (
    als_cpd,
    init_weights,
    predict_score,
    process_weights,
    hess_cov_estimation,
)
from ..prob_functions import (
    predict_std,
    init_beta_e,
    init_gamma_w,
    update_beta_e, 
    update_gamma_w,
)

class LaplaceCPR(RegressorMixin, BaseEstimator):
    def __init__(
        self, 
        rank: int = 1, 
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        beta_e: Optional[float] = 1.0,
        gamma_w: Optional[float] = 1e-3,
        pd_mode: str = 'lla', 
        hess_type: Optional[str] = 'last',
        hess_th: Optional[float] = None,
        seed: Optional[int] = None,
        n_epoch_vi: int = 1,
        pd_samples: int = 30, 
        pd_sample_seed: Optional[int] = None, 
        beta_e_samples: int = 10, 
        tracker: Optional[object] = None,
    ):
        self.rank = rank 
        self.fmap = fmap 
        self.m_order = m_order 
        self.n_epoch = n_epoch 
        self.pd_mode = pd_mode
        self.hess_type = hess_type
        self.hess_th = hess_th
        self.seed = seed
        self.n_epoch_vi = n_epoch_vi
        self.pd_samples = pd_samples
        self.pd_sample_seed = pd_sample_seed
        self.beta_e_samples = beta_e_samples
        self.tracker = tracker
        self._quant = False
        self._qbase = None
        self._init_type = None
        self._beta_e_sample_seed = seed
        # Prepare local nonlinear map:
        self._fmap, self._dtype = prepare_fmap(
            fmap, m_order, self._quant
        )
        # Initialize precision:
        self.beta_e, self.upd_beta_e, self.cn, self.dn = init_beta_e(beta_e)
        self.gamma_w, self.upd_gamma_w, self.an, self.bn = init_gamma_w(gamma_w)

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        # Data checks:
        X, y = check_X_y(X, y)
        X, y = jnp.array(X), jnp.array(y)
        # Init model parameters:
        self.w_ten, self.kd = init_weights(
            self.m_order, 
            self.rank, 
            X.shape[-1], 
            self._qbase, 
            self._init_type, 
            self.seed, 
            self._dtype
        )
        self.w_shape = self.w_ten.shape
        # VI training loop:
        for ep_g in range(self.n_epoch_vi):
            self._update_w(X, y, xy_test) # Update weights mean and cov.;
            if not self.upd_beta_e and not self.upd_gamma_w:
                break 
            if self.upd_gamma_w:
                self._update_gamma_w() # Update weights prior precision; 
            if self.upd_beta_e:
                self._update_beta_e(X, y) # Update noise precision;
        self.is_fitted_ = True
        return self
    
    def predict(self, X, return_std=False, std_use_noise=True):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        pred_mean = predict_score(X, self.kd, self.w_ten, self._fmap)
        if return_std:
            pred_std = self._predict_std(X, std_use_noise)
            return pred_mean, pred_std
        return pred_mean
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
    
    def _update_w(self, X, y, xy_test: Optional[tuple] = None):
        if xy_test: raise NotImplementedError()

        w_vec = ten3tovec(self.w_ten)
        w_vec = als_cpd(
            w_vec, self.kd, self.w_shape, 
            X, y, self._fmap, self.n_epoch, 
            self.gamma_w, self.beta_e, self.tracker
        )
        self.w_ten = vec2ten3(w_vec, *self.w_shape)
        self.w_ten, self.w_shape = process_weights(self.w_ten)
        # Hessian/Covariance/Cholesky Evaluation:
        self.w_hess, self.w_cov, self.w_cholesky = hess_cov_estimation(
            self.w_ten, self.kd, X, y, self._fmap, 
            self.gamma_w, self.beta_e, self.hess_type, self.hess_th
        )

    def _update_beta_e(self, X, y):
        if self._beta_e_sample_seed:
            self._beta_e_sample_seed += 1
        self.cn, self.dn, self.beta_e = update_beta_e(
            self.cn, self.dn, 
            self.w_ten, self.w_cholesky, 
            self.kd, X, y, self._fmap,
            self.pd_mode, self.beta_e_samples, 
            self._beta_e_sample_seed,
        )

    def _update_gamma_w(self):
        self.an, self.bn, self.gamma_w = update_gamma_w(
            self.an, self.bn, 
            self.w_cholesky, self.w_ten
        )

    def _predict_std(self, X, std_use_noise): 
        beta_e = self.beta_e if std_use_noise else None
        return predict_std(
            self.w_ten, self.w_cholesky, self.kd, 
            X, self._fmap, beta_e,
            self.pd_mode, self.pd_samples,
            self.pd_sample_seed,
        ) 
