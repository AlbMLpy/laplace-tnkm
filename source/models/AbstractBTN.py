from typing import Optional
from functools import partial
from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import jit, jacrev

from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted

from ..features import Feature, PPFeature, prepare_fmap
from ..model_functionality import predict_score

class AbstractBTN(ABC, RegressorMixin, BaseEstimator): 
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
    ):
        self.rank = rank # Rank of CPD - weights parameterization;
        self.fmap = fmap # Local non-linear feature map;
        self.m_order = m_order # I or M - dimensionality of a local map (paper);
        self.n_epoch = n_epoch # The number of full sweeps through the data;
        self.beta_e = beta_e # Precision of irreducible noise term e: y = f(x) + e;
        self.beta_w = beta_w # Precision for weights diagonal prior;
        self.seed = seed # Reproducibility parameter;
        self.opt_params = opt_params # Defines optimization solver and its parameters;
        self.n_epoch_global = n_epoch_global # Defines #sweeps between dist. changes;
        self.pd_nsamples = pred_dist_n_samples # For pred. dist. estimation #samples;
        self.beta_e_n_samples = beta_e_n_samples # For beta_e update #samples;
        # Initialize precisions if needed:
        self.upd_beta_e = beta_e is None
        self.upd_beta_w = beta_w is None
        if self.upd_beta_e:
            self.cn = self.dn = 0.1 ### Not clear ###
            self.beta_e = self.cn / self.dn 
        if self.upd_beta_w:
            self.an = self.bn = 0.1 ### Not clear ###
            self.beta_w = self.an / self.bn
        ### Future ###
        self._quantized = False ### Non-quantized ###
        self.kd = 1 ### Non-quantized ###

        # Prepare local nonlinear map:
        self._fmap, self._dtype = prepare_fmap(fmap, m_order, self._quantized)

    @abstractmethod
    def fit(self, X, y, xy_test: Optional[tuple] = None):
        """ 
        Should generate:fmap
        X, y = check_X_y(X, y)
        self.kd, self.w_mean, self.w_cov = None, None, None
        self.is_fitted_ = True
        return self
        """
        pass
    
    @abstractmethod
    def _w_sample(self, w_mean_vec):
        pass
    
    @abstractmethod
    def _linearized(self, g):
        pass

    def _update_beta_e(self, X, y):
        mean_train_err = 0.0
        for _ in range(self.beta_e_n_samples):
            weights = self._w_sample(self.w_mean.reshape(-1)).reshape(self.w_shape)
            scores = predict_score(X, self.kd, weights, self._fmap) 
            mean_train_err += jnp.sum((y - scores)**2)
        mean_train_err /= 2 * self.beta_e_n_samples
        self.cn += 0.5 * X.shape[0]
        self.dn += mean_train_err
        self.beta_e = self.cn / max(self.dn, 1e-8)

    def _update_beta_w(self):
        w_cov_diag = self.w_cov if self.w_cov.ndim == 1 else self.w_cov.diagonal()
        self.an += 0.5 * self.w_mean.size
        self.bn += (self.w_mean * self.w_mean).sum() + w_cov_diag.sum()
        self.beta_w = self.an / max(self.bn, 1e-8)
    
    def _predict_mean(self, X):
        return predict_score(X, self.kd, self.w_mean, self._fmap)

    def predict(self, X):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        pred_mean = self._predict_mean(X)
        return pred_mean
    
    def _predict_std(self, X, std_mode: Optional[str] = None):
        std_err = 1 / jnp.sqrt(self.beta_e)
        if std_mode == 'sampling':
            w_mean_vec = self.w_mean.reshape(-1)
            preds = []
            for _ in range(self.pd_nsamples):
                w_sample = self._w_sample(w_mean_vec)
                prediction = predict_score(X, self.kd, w_sample.reshape(self.w_shape), self._fmap)
                preds.append(prediction[:, None])
            pred_std = jnp.std(jnp.hstack(preds), axis=1)
        elif std_mode == 'linearize':
            g_jac = jit(jacrev(predict_score, argnums=2), static_argnums=(3,))
            g = g_jac(X, self.kd, self.w_mean, self._fmap).reshape(X.shape[0], -1)
            pred_std = self._linearized(g)
        else:
            raise ValueError(f'Bad std_sampling = "{std_mode}". See docs.')
        return pred_std + std_err
    
    def predict_std(self, X, std_mode: Optional[str] = None):
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        pred_std = self._predict_std(X, std_mode)
        return pred_std
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
