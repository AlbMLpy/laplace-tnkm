from typing import Optional
from functools import partial

import jax
import optax
import numpy as np
from jax import jit
from flax import struct
import jax.numpy as jnp
from sklearn.utils.validation import check_X_y

from ..evaluation import l2_loss_kl
from .AbstractBTN import AbstractBTN
from ..matrix_operations import vec2ten3
from ..optimization import std_transform
from ..features import Feature, PPFeature
@struct.dataclass
class MFBTNParams:
    m: jnp.ndarray  # mean vector
    p: jnp.ndarray  # log-std vector (pre-transformed)

class MeanFieldBTN(AbstractBTN):
    """
    Mean-Field Variational Inference for Bayesian Tensor Networks.
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
        opt_params: Optional[dict] = None,
        n_epoch_vi: int = 1,
        pd_samples: int = 30,
        beta_e_samples: int = 10,
        tracker: Optional[object] = None,
        n_loss_samples: int = 30, 
    ):
        super().__init__(
            rank, fmap, m_order, n_epoch, beta_e, gamma_w, seed, 
            opt_params, n_epoch_vi, pd_samples, beta_e_samples, tracker,
        )
        self.n_loss_samples = n_loss_samples
        self._loss = l2_loss_kl_mf
        self._loss_key = jax.random.PRNGKey(np.random.RandomState(seed).randint(1e18))

    def fit(self, X, y, xy_test: Optional[tuple] = None):
        X, y = check_X_y(X, y)
        X, y = jnp.array(X), jnp.array(y)
        self.w_shape = (X.shape[-1], self.m_order, self.rank)
        self.params = self._init_params()
        self.loss_list = []
        # VI training loop:
        for _ in range(self.n_epoch_vi):
            self._loss_key, subkey = jax.random.split(self._loss_key)
            self._update_w(X, y, subkey, xy_test)
            if self.upd_gamma_w:
                self._update_gamma_w()
            if self.upd_beta_e:
                self._update_beta_e(X, y)
            if not self.upd_beta_e and not self.upd_gamma_w:
                break
        self.is_fitted_ = True
        return self

    def _init_params(self):
        key = jax.random.PRNGKey(np.random.RandomState(self.seed).randint(1e18))
        key, subkey = jax.random.split(key)
        return MFBTNParams(
            0.5*jax.random.normal(key, (np.prod(self.w_shape),)),
            0.5*jax.random.normal(subkey, (np.prod(self.w_shape),)),
        )

    def _update_w(self, X, y, key, xy_test: Optional[tuple] = None):
        if xy_test: raise NotImplementedError("Test-time updates not implemented.")
        self.params, loss_list = train(
            self.params, X, y, key, self.w_shape, self.kd, self._fmap, 
            self.gamma_w, self.beta_e, self.n_epoch, self.n_loss_samples, 
            self.opt_params['train_mode'], self.opt_params['lr'], self._loss,
        )
        self.loss_list.extend(loss_list)
        self._postprocess()

    def _postprocess(self):
        self.w_mean = vec2ten3(self.params.m, *self.w_shape)
        self.w_cholesky = jnp.diag(std_transform(self.params.p))

def train(
    params, X, y, key, w_shape, kd, fmap, gamma_w, beta_e, 
    n_epochs, n_samples, opt_mode, lr, loss_fn
):
    optimizer = make_optimizer(opt_mode, lr)
    opt_state = optimizer.init(params)
    loss_list = []
    for _ in range(n_epochs):
        params, opt_state, loss, key = update_step(
            params, opt_state, X, y, key, w_shape, kd, fmap,
            gamma_w, beta_e, n_samples, optimizer, loss_fn
        )
        loss_list.append(loss)
    return params, loss_list

def make_optimizer(opt_mode='adam', lr=1e-3):
    if opt_mode == 'adam':
        return optax.adam(lr)
    else:
        return optax.sgd(lr)

@partial(jit, static_argnums=[5, 7, 10, 11, 12])
def update_step(params, opt_state, x, y, key, w_shape, kd, fmap, gamma_w, beta_e, n_samples, optimizer, loss_fn):
    loss_grad_fn = jax.value_and_grad(loss_fn)
    loss_key, new_key = jax.random.split(key)
    loss, grads = loss_grad_fn(
        params, w_shape, kd, x, y, fmap, gamma_w, beta_e, loss_key, n_samples
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss, new_key

def l2_loss_kl_mf(params, w_shape, kd, x, y, fmap, gamma_w, beta_e, key, n_samples):
    w_mean_vec, w_std_vec = params.m, std_transform(params.p)
    return l2_loss_kl(
        w_mean_vec, w_std_vec, w_shape, kd, x, y, 
        fmap, gamma_w, beta_e, key, n_samples,
    )
