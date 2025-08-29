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
    """
    Bayesian Tensor Network Kernel Machine (LA-TNKM) that 
    uses a (linearized) Laplace approximation for Bayesian inference.

    Currently, the implementation is restricted to using the canonical
    polyadic decomposition (CPD).

    Parameters
    ----------
    rank : int, default=1
        Rank of the CPD weights tensor.

    fmap : Feature, default=PPFeature()
        Nonlinear data mapping to get new features. 
        Other options can be: [PPNFeature, RBFFeature].

    m_order : int, default=2
        The number of newly generated features per data feature, x_d.
        E.g., in case of 'PPFeature', m_order is the order of the polynomial.

    n_epoch : int, default=1
        The number of full ALS updates (sweeps).

    beta_e : float, optional, default=1.0
        Noise precision hyperparameter. If 'beta_e=None' the hyperparameter 
        is evaluated with variational inference.

    gamma_w : float, optional, default=1e-3
        Prior precision hyperparameter. If 'gamma_w=None' the hyperparameter 
        is evaluated with variational inference. Note that, this hyperparameter 
        is sensitive and can lead to zero solution, w_ten = 0, very quickly.

    pd_mode : str, default='lla'
        Type of the predictive distribution: 'lla' refers to Linearized Laplace
        Approximation (LLA), 'la' refers to Laplace Approximation (LA). 

    hess_type : str, optioinal, default='last'
        Type of Hessian approximation: 'full' represents full Hessian mode, 
        'gauss_newton' represents generalized Gauss-Newton, 'block' refers to 
        block-diagonal approximation, 'mf' represents diagonal approximation 
        and 'last' means being Bayesian only to the last CPD core.

    hess_th : float, optional, default=None
        Thresholding hyperparameter to eliminate problematic eigenvalues of the 
        Hessian or its approximation. If 'hess_th=None' then no thresholding 
        is applied.

    seed : int, optional, default=None
        Determines random number generation used to initialize 
        the model parameters. Pass an int for reproducible results.

    n_epoch_vi : int, default=1
        The number of variational inference updates of 
        the model parameters: w_ten, beta_e, gamma_w.

    pd_samples : int, default=30
        The number samples to estimate the expectation
        (predictive distribution) with Monte Carlo sampling.

    pd_sample_seed : int, optional, default=None
        Determines random number generation used by Monte Carlo sampler to
        estimate the expectation. Pass an int for reproducible results.

    beta_e_samples : int, default=10
        Determines the number of samples used to estimate the noise precision.

    tracker : Tracker object, optional, default=None
        This object can be used to gather useful statistics during training.
        If 'tracker=None' then no tracking is used. 

    Attributes
    ----------
    w_ten : array-like
        Array containing the CPD weights with the following
        shape: (d_dim, m_order, rank).

    w_cholesky : array-like
        Array containing Cholesky factor matrix of the corresponding 
        covariance matrix (inverse of the Hessian).

    Examples
    --------
    >>> from sklearn.datasets import make_friedman2
    >>> from source.models.LaplaceCPR import LaplaceCPR
    >>> from source.features import PPNFeature
    >>> from source.exp_functions import model_factory
    >>> X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
    >>> la_tnkm = model_factory(
    ...     LaplaceCPR, 
    ...     dict(rank=2, fmap=PPNFeature(), m_order=16, 
    ...          n_epoch=5, seed=0, pd_sample_seed=1), 
    ...     scaler='std',
    ... ).fit(X, y)
    >>> la_tnkm.score(X, y)
    0.2098...
    >>> la_tnkm.predict(X[:2,:], return_std=True)
    (
        Array([513.22484902, -30.7807934 ], dtype=float64), 
        Array([1.44623227, 1.04189542], dtype=float64)
    )
    """

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
        """
        Fit Bayesian tensor network kernel machine.

        Parameters
        ----------
        X : array-like of shape (n_samples, d_dim)
            Training data matrix.

        y : array-like of shape (n_samples,)
            Target values.

        xy_test : tuple, optional, default=None
            Test dataset.

        Returns
        -------
        self : object
            LaplaceCPR class instance.
        """
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
        """
        Predict using the Bayesian tensor network kernel machine.

        In addition to the mean of the predictive distribution, optionally also
        returns its standard deviation ('return_std=True') and adds 
        the Gaussian additive noise std ('std_use_noise=True').

        Parameters
        ----------
        X : array-like of shape (n_samples, d_dim) 
            Query points where the LA-TNKM is evaluated.

        return_std : bool, default=False
            If True, the standard-deviation of the predictive distribution at
            the query points is returned along with the mean.

        std_use_noise : bool, default=True
            If True, adds Gaussian noise defined by the beta_e hyperparameter.

        Returns
        -------
        y_mean : ndarray of shape (n_samples,)
            Mean of predictive distribution at query points.

        y_std : ndarray of shape (n_samples,), optional
            Standard deviation of predictive distribution at query points.
            Only returned when `return_std` is True.
        """
        X = check_array(X)
        check_is_fitted(self, 'is_fitted_')
        pred_mean = predict_score(X, self.kd, self.w_ten, self._fmap)
        if return_std:
            pred_std = self._predict_std(X, std_use_noise)
            return pred_mean, pred_std
        return pred_mean
    
    def score(self, X, y):
        """
        Return coefficient of determination, R^2, on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, d_dim)
            Test samples.

        y : array-like of shape (n_samples,)
            True values for X.

        Returns
        -------
        score : float
            R^2 of self.predict(X) w.r.t. y.
        """
        return r2_score(y, self.predict(X))
    
    def _update_w(self, X, y, xy_test: Optional[tuple] = None):
        """ 
        Update CPD weights (mean) and covariance matrix (Cholesky factor). 
        """
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
        """ Update Gaussian additive noise precision, beta_e. """
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
        """ Update Gaussian prior weights precision, gamma_w. """
        self.an, self.bn, self.gamma_w = update_gamma_w(
            self.an, self.bn, 
            self.w_cholesky, self.w_ten
        )

    def _predict_std(self, X, std_use_noise): 
        """ Compute standard deviation of the predictive distribution."""
        beta_e = self.beta_e if std_use_noise else None
        return predict_std(
            self.w_ten, self.w_cholesky, self.kd, 
            X, self._fmap, beta_e,
            self.pd_mode, self.pd_samples,
            self.pd_sample_seed,
        ) 
