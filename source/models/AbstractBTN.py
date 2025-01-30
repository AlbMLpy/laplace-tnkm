from typing import Optional
from functools import partial

import jax.numpy as jnp

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..features import (
    Feature, PPFeature, 
    ppf_q2, ff_q2,
    pure_poli_features, gaussian_kernel_features,
)

class AbstractBTN(RegressorMixin, BaseEstimator): # Non-quantized
    def __init__(
        self, 
        rank: int = 1, 
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        std_err: float = 0.01,
        std_w: float = 0.01,
        seed: Optional[int] = None,
        opt_params: dict = {'train_mode': 'GD', 'lr': 1e-3}

    ):
        self.rank = rank # Rank of CPD - weights parameterization;
        self.fmap = fmap # Local non-linear feature map;
        self.m_order = m_order # In paper can be I or M - dimensionality of a local map;
        self.n_epoch = n_epoch # The number of full sweeps through the data;
        self.std_err = std_err # Standard deviation of irreducible noise term e: y = f(x) + e;
        self.std_w = std_w # Standard deviation for weights diagonal prior;
        self.seed = seed # Reproducibility parameter;
        self.opt_params = opt_params # Defines optimization solver and its parameters

        self._quantized = False
        self._dtype = None

    def _prepare_fmap(self):
        if self._quantized:
            if self.fmap.name == 'ppf':
                self._dtype = jnp.float64
                return ppf_q2
            elif self.fmap.name == 'ff':
                self._dtype = jnp.complex128
                return partial(
                    ff_q2, 
                    m_order=self.m_order, 
                    k_d=int(jnp.log2(self.m_order)), 
                    p_scale=self.fmap.p_scale,
                )
            else:
                raise ValueError(f'Bad feature_map = "{self.fmap}". See docs.')
        else:
            if self.fmap.name == 'ppf':
                self._dtype = jnp.float64
                return partial(pure_poli_features, order=self.m_order)
            elif self.fmap.name == 'rbff':
                self._dtype = jnp.float64
                return partial(
                    gaussian_kernel_features, 
                    order=self.m_order, 
                    lscale=self.fmap.l_scale, 
                )
            else:
                raise ValueError(f'Bad feature_map = "{self.fmap}". See docs.')

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        self.kd = None
        self.w_mean = None
        self.w_cov = None
        self.is_fitted_ = True
        return self
    
    def _predict_mean(self, X):
        pass

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        pred_mean = self._predict_mean(X)
        return pred_mean
    
    def _predict_std(self, X, std_mode: Optional[str] = None):
        pass
    
    def predict_std(self, X, std_mode: Optional[str] = None):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        pred_std = self._predict_std(X, std_mode)
        return pred_std
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
