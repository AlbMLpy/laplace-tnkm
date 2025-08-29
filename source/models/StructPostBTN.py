from typing import Optional

import jax
import numpy as np
from flax import struct
import jax.numpy as jnp

from ..evaluation import l2_loss_kl
from .MeanFieldBTN import MeanFieldBTN
from ..optimization import std_transform
from ..features import Feature, PPFeature
from ..matrix_operations import cpd_transform_vec, vec2ten3


@struct.dataclass
class SPBTNParams:
    md: jnp.ndarray # mean vector - data_dim
    mi: jnp.ndarray # mean vector - local feature dim
    mr: jnp.ndarray # mean vector - CPD rank
    pd: jnp.ndarray # log-std vector - data_dim
    pi: jnp.ndarray # log-std vector - local feature dim
    pr: jnp.ndarray # log-std vector - CPD rank

class StructPostBTN(MeanFieldBTN):
    """
    Structured Posteriors Variational Inference for Bayesian Tensor Networks
    (SP-BTN).

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
        The number gradient descent epochs.

    beta_e : float, optional, default=1.0
        Noise precision hyperparameter. If 'beta_e=None' the hyperparameter 
        is evaluated with variational inference.

    gamma_w : float, optional, default=1.0
        Prior precision hyperparameter. If 'gamma_w=None' the hyperparameter 
        is evaluated with variational inference. 

    seed : int, optional, default=None
        Determines random number generation used to initialize 
        the model parameters. Pass an int for reproducible results.

    opt_params : dict, optional, None
        Optimizer parameters of the form: {'train_mode': mode, 'lr': float}, 
        where mode='adam', 'sgd'. If None, then the default config is used:
        {'train_mode': 'sgd', 'lr': 1e-3}.

    n_epoch_vi : int, default=1
        The number of variational inference updates of 
        the model parameters: w_ten, beta_e, gamma_w.

    pd_samples : int, default=30
        The number samples to estimate the expectation
        (predictive distribution) with Monte Carlo sampling.

    beta_e_samples : int, default=10
        Determines the number of samples used to estimate the noise precision.

    tracker : Tracker object, optional, default=None
        This object can be used to gather useful statistics during training.
        If 'tracker=None' then no tracking is used. 

    n_loss_samples : int, default=30
        The number of samples used to estimate the variational loss.

    m_rank : int, default=1
        CPD rank, M, for the mean weights tensor (represented as a CPD itself).

    Attributes
    ----------
    params : SPBTNParams
        Dataclass containing the following attributes: 
        - md: array - mean vector for d_dim;
        - mi: array - mean vector for local feature dim;
        - mr: array - mean vector for CPD rank, R;
        - pd: array - log-std vector for d_dim;
        - pi: array - log-std vector for local feature dim;
        - pr: array - log-std vector for CPD rank, R;

    References
    ---------- 
    - "Bayesian Tensor Networks with Structured Posteriors", 
        Kriton Konstantinidis, Yao Lei Xu, Danilo P. Mandic. 2021.
    """
    
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
        m_rank: int = 1, 
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, beta_e, gamma_w, seed, 
            opt_params, n_epoch_vi, pd_samples, beta_e_samples, tracker,
            n_loss_samples,
        )
        self.m_rank = m_rank
        self._loss = l2_loss_kl_sp

    def _init_params(self):
        key = jax.random.PRNGKey(
            np.random.RandomState(self.seed).randint(1e18)
        )
        keys = jax.random.split(key, num=6)
        shapes_m = [(mk, self.m_rank) for mk in self.w_shape]
        shapes_p = [(mk,) for mk in self.w_shape]
        shapes = shapes_m + shapes_p
        return SPBTNParams(
            *[0.5*jax.random.normal(key, shape) 
              for key, shape in zip(keys, shapes)]
        )
    
    def _postprocess(self):
        w_mean = cpd_transform_vec(
            self.params.md, self.params.mi, self.params.mr
        )
        w_std = std_transform_sp(
            self.params.pd, self.params.pi, self.params.pr
        )
        self.w_mean = vec2ten3(w_mean, *self.w_shape)
        self.w_cholesky = jnp.diag(w_std) # Not efficient!
    
def std_transform_sp(pd, pi, pr):
    """ Standard deviation transformation in SP-BTN case. """
    pd, pi, pr = map(std_transform, (pd, pi, pr))
    return jnp.kron(pd, jnp.kron(pi, pr))

def l2_loss_kl_sp(
    params, 
    w_shape, 
    kd, 
    x, 
    y, 
    fmap, 
    gamma_w, 
    beta_e, 
    key, 
    n_samples
):
    """ Compute variational loss function in SP-BTN case. """
    w_mean_vec = cpd_transform_vec(params.md, params.mi, params.mr)
    w_std_vec = std_transform_sp(params.pd, params.pi, params.pr)
    return l2_loss_kl(
        w_mean_vec, w_std_vec, w_shape, kd, x, y, 
        fmap, gamma_w, beta_e, key, n_samples,
    )
