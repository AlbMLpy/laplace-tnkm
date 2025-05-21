from typing import Optional, Callable

import jax.numpy as jnp
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from ..cpr import cpr
from ..model_functionality import predict_score
from ..features import Feature, PPFeature, prepare_fmap

class QCPR(RegressorMixin, BaseEstimator):
    def __init__(
        self, 
        rank: int = 1, 
        fmap: Feature = PPFeature(), 
        m_order: int = 2,
        n_epoch: int = 1, 
        alpha: float = 1.0, 
        seed: Optional[int] = None,
        quant: bool = False,
        callback: Optional[Callable] = None,
    ):
        self.rank = rank
        self.fmap = fmap
        self.m_order = m_order
        self.init_type = None
        self.n_epoch = n_epoch
        self.alpha = alpha
        self.seed = seed
        self.quant = quant
        self.callback = callback
        self._dtype = None
        self.ww_reg = False
        self.pinv = False
        # Prepare local nonlinear map:
        self._fmap, self._dtype = prepare_fmap(fmap, m_order, self.quant)

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        self.weights_, self.kd_ = cpr(
            jnp.array(X), jnp.array(y), self.quant, self.m_order, self._fmap, 
            self.rank, self.init_type, self.n_epoch, self.alpha, self.seed,
            self._dtype, xy_test, self.callback, pinv=self.pinv, ww_reg=self.ww_reg
        )
        self.is_fitted_ = True
        return self
    
    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        return predict_score(X, self.kd_, self.weights_, self._fmap)
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
