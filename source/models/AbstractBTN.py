from typing import Optional
from abc import ABC, abstractmethod

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ..features import Feature, PPFeature, prepare_fmap
from ..model_functionality import predict_score

from ..prob_functions import (
    predict_std,
    init_beta_e,
    init_gamma_w,
    update_beta_e, 
    update_gamma_w,
)

class AbstractBTN(ABC, RegressorMixin, BaseEstimator): 
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
    ):
        self.rank = rank
        self.fmap = fmap
        self.m_order = m_order
        self.n_epoch = n_epoch
        self.beta_e = beta_e 
        self.gamma_w = gamma_w
        self.seed = seed
        self.opt_params = opt_params 
        self.n_epoch_vi = n_epoch_vi
        self.pd_samples = pd_samples
        self.beta_e_samples = beta_e_samples
        self.tracker = tracker
        self.kd = 1
        self.pd_mode = 'la'
        self._quant = False
        self._pd_sample_seed = seed if seed is None else seed + 1
        self._beta_e_sample_seed = seed if seed is None else seed + 2
        # Prepare local nonlinear map:
        self._fmap, self._dtype = prepare_fmap(fmap, m_order, self._quant)
        # Initialize precision:
        self.beta_e, self.upd_beta_e, self.cn, self.dn = init_beta_e(beta_e)
        self.gamma_w, self.upd_gamma_w, self.an, self.bn = init_gamma_w(gamma_w)

    @abstractmethod
    def fit(self, X, y, xy_test: Optional[tuple] = None):
        pass

    def predict(self, X, return_std=False, std_use_noise=True):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        pred_mean = self._predict_mean(X)
        if return_std:
            pred_std = self._predict_std(X, std_use_noise)
            return pred_mean, pred_std
        return pred_mean
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
    
    def _predict_mean(self, X):
        return predict_score(X, self.kd, self.w_mean, self._fmap)

    def _update_beta_e(self, X, y):
        if self._beta_e_sample_seed:
            self._beta_e_sample_seed += 1
        self.cn, self.dn, self.beta_e = update_beta_e(
            self.cn, self.dn, 
            self.w_mean, self.w_cholesky, 
            self.kd, X, y, self._fmap,
            self.pd_mode, self.beta_e_samples, 
            self._beta_e_sample_seed,
        )

    def _update_gamma_w(self):
        self.an, self.bn, self.gamma_w = update_gamma_w(
            self.an, self.bn, self.w_cholesky, self.w_mean
        )
    
    def _predict_std(self, X, std_use_noise):
        beta_e = self.beta_e if std_use_noise else None
        return predict_std(
            self.w_mean, self.w_cholesky, self.kd, 
            X, self._fmap, beta_e,
            self.pd_mode, self.pd_samples,
            self._pd_sample_seed,
        )
