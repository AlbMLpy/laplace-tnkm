from functools import partial
from typing import Optional, Callable

import jax
import optax
import numpy as np
from flax import nnx
import jax.numpy as jnp
from sklearn.metrics import r2_score
from laplax.eval.metrics import nll_gaussian
from laplax.util.loader import input_target_split
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, RegressorMixin
from laplax.curv import create_ggn_mv, create_posterior_fn
from laplax.eval import evaluate_for_given_prior_arguments
from laplax.eval.pushforward import (
    lin_setup,
    nonlin_setup,
    lin_pred_std,
    lin_pred_mean,
    nonlin_pred_mean,
    nonlin_pred_std,
    set_lin_pushforward,
    set_nonlin_pushforward,
)

from ..prob_functions import init_beta_e

class LaplaceBNN(RegressorMixin, BaseEstimator):
    def __init__(
        self, 
        nn_arch,
        opt,
        batch_size: int,
        n_epoch: int,  
        beta_e: Optional[float] = 1.0,
        gamma_w: float = 1.0,
        seed: Optional[int] = None,
        *,
        curv: str = 'full', # 'full', 'diagonal'
        linearized: bool = True,
        shuffle: bool = True,
        verbose: bool = False,
    ):
        self.nn_arch = nn_arch
        self.opt = opt
        self.optimizer = nnx.Optimizer(self.nn_arch, self.opt)
        self.batch_size = batch_size
        self.n_epoch = n_epoch
        self.beta_e = beta_e
        self.gamma_w = gamma_w
        self.seed = np.random.randint(1e10) if seed is None else seed
        self.curv = curv
        self.linearized = linearized
        self.shuffle = shuffle
        self.verbose = verbose
        # Internal parameters:
        self._loss_fn = 'mse'
        self._n_samples = 1000
        self._valid_size = 0.05
        self._hyper_lr = 1e-3
        self._n_clbr_epochs = 30
        self.rank = None
        # Initialize precision:
        self.beta_e, self.upd_beta_e, _, _ = init_beta_e(beta_e)
        self.beta_e, self.gamma_w = map(
            jnp.asarray, [self.beta_e, self.gamma_w])

    def fit(self, X, y):
        # Data preparation:
        train_loader, train_batch = self._prepare_data(X, y)
        # Compute Mean:
        train_model(
            train_loader, self.nn_arch, 
            self.optimizer, self.n_epoch, self.verbose,
        )
        # Create GGN:
        graph_def, params = nnx.split(self.nn_arch)
        def model_fn(input, params):
            return nnx.call((graph_def, params))(input)[0]
        posterior_fn = self._get_posterior_fn(train_batch, params, model_fn)
        if self.linearized:
            self._linearized(model_fn, params, posterior_fn)
        else:
            self._non_linearized(model_fn, params, posterior_fn)

    def predict(self, X, return_std=False, std_use_noise=True):
        pred = jax.vmap(self.prob_predictive)(X)
        pred_mean = pred['pred_mean'][:, 0]
        pred_std = pred['pred_std'][:, 0]
        if return_std:
            if std_use_noise: 
                pred_std = pred_std + 1/jnp.sqrt(self.beta_e)
            return pred_mean, pred_std
        return pred_mean
    
    def _prepare_data(self, X, y):
        if self.upd_beta_e:
            X, X_valid, y, y_valid = train_test_split(
                X, y, test_size=self._valid_size, random_state=self.seed,
            )
            X, X_valid, y, y_valid = map(jnp.array, [X, X_valid, y, y_valid])
            self.clbr_batch = {'input': X_valid, 'target': y_valid}
        train_loader = DataLoader(X, y, self.batch_size, shuffle=self.shuffle)
        train_batch = dict(input=X, target=y)
        return train_loader, train_batch
    
    def _get_posterior_fn(self, train_batch, params, model_fn):
        ggn_mv = create_ggn_mv(
            model_fn, params, train_batch, loss_fn=self._loss_fn)
        return create_posterior_fn(
            curv_type=self.curv, mv=ggn_mv, layout=params)

    def _linearized(self, model_fn, params, posterior_fn):
        set_prob_predictive = partial(
            set_lin_pushforward,
            model_fn=model_fn,
            mean_params=params,
            posterior_fn=posterior_fn,
            pushforward_fns=[lin_setup, lin_pred_mean, lin_pred_std],
            key=jax.random.key(self.seed),
            n_samples=self._n_samples,
        )
        # Update hyperparameters:
        self.prior_args = self._train_hyper(set_prob_predictive)
        self.beta_e = 1 / self.prior_args['sigma_squared']
        self.gamma_w = self.prior_args['prior_prec']
        # Create predictive model:
        self.prob_predictive = set_prob_predictive(
            prior_arguments=self.prior_args,
        )

    def _non_linearized(self, model_fn, params, posterior_fn):
        set_nonlin_prob_predictive = partial(
            set_nonlin_pushforward,
            model_fn=model_fn,
            mean_params=params,
            posterior_fn=posterior_fn,
            pushforward_fns=[nonlin_setup, nonlin_pred_mean, nonlin_pred_std],
            key=jax.random.key(self.seed),
            num_samples=self._n_samples,
        )
        self.prob_predictive = set_nonlin_prob_predictive(
            prior_arguments=self.prior_args,
        )

    def _prepare_prior_args_train(self):
        prior_args = {}
        if self.upd_beta_e:
            prior_args['sigma_squared'] = 1 / self.beta_e
        prior_args['prior_prec'] = self.gamma_w
        return prior_args

    def _train_hyper(self, set_prob_predictive):
        prior_args = self._prepare_prior_args_train()
        if self.upd_beta_e:
            @jax.jit
            def nll_objective(prior_arguments, batch):
                return evaluate_for_given_prior_arguments(
                    prior_arguments=prior_arguments,
                    data=batch,
                    set_prob_predictive=set_prob_predictive,
                    metric=nll_gaussian,
                )
            objective = nll_objective
            prior_args = train_hyper(
                objective, 
                prior_args, 
                self.clbr_batch['input'], 
                self.clbr_batch['target'], 
                lr=self._hyper_lr, 
                n_clbr_epochs=self._n_clbr_epochs
            )
        if not self.upd_beta_e:
            prior_args['sigma_squared'] = 1 / self.beta_e
        prior_args['prior_prec'] = self.gamma_w
        return prior_args
    
    def score(self, X, y):
        return r2_score(y, self.predict(X))
    
def train_hyper(
    objective: dict[Callable], 
    prior_arguments: dict, 
    X_valid, 
    y_valid, 
    lr=1e-3, 
    n_clbr_epochs=10,
    batch_size=16,
):
    # Set optimizer
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(prior_arguments)
    valid_loader = DataLoader(X_valid, y_valid, batch_size=batch_size)
    # Transform prior arguments, so we can optimize over all reals
    prior_arguments = jax.tree.map(jnp.log, prior_arguments)
    # Optimize prior arguments
    for _ in range(n_clbr_epochs):
        epoch_vals = []
        for batch in valid_loader:
            val, grads = jax.value_and_grad(
                lambda p: objective(
                    jax.tree.map(jnp.exp, p),
                    input_target_split(batch),  # noqa: B023
                )
            )(prior_arguments)
            # Update the parameters using the optimizer
            updates, opt_state = optimizer.update(grads, opt_state)
            prior_arguments = optax.apply_updates(prior_arguments, updates)
            epoch_vals.append(val)
    # Transform prior arguments back
    prior_arguments = jax.tree.map(jnp.exp, prior_arguments)
    return prior_arguments

@nnx.jit
def train_step(model, optimizer, x, y):
    def loss_fn(model):
        y_pred = model(x)  # Call methods directly
        return jnp.sum((y_pred - y) ** 2)
    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)  # Inplace updates
    return loss

def train_model(train_loader, model, optimizer, n_epochs, verbose=False):
    for epoch in range(n_epochs):
        for x_tr, y_tr in train_loader:
            loss = train_step(model, optimizer, x_tr, y_tr)
        if epoch % 100 == 0 and verbose:
            print(f"[epoch {epoch}]: loss: {loss:.4f}")
    if verbose:
        print(f"Final loss: {loss:.4f}")
    return model
        

class SimpleNN(nnx.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, rngs):
        self.linear1 = nnx.Linear(in_channels, hidden_channels, rngs=rngs)
        self.linear2 = nnx.Linear(hidden_channels, out_channels, rngs=rngs)

    def __call__(self, x):
        x = self.linear2(nnx.tanh(self.linear1(x)))
        return x


class DataLoader:
    """Simple dataloader."""

    def __init__(self, X, y, batch_size, *, shuffle=True) -> None:
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset_size = X.shape[0]
        self.indices = np.arange(self.dataset_size)
        self.rng = np.random.default_rng(seed=0)

    def __iter__(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.dataset_size:
            raise StopIteration
        start_idx = self.current_idx
        end_idx = start_idx + self.batch_size
        batch_indices = self.indices[start_idx:end_idx]
        self.current_idx = end_idx
        return self.X[batch_indices], self.y[batch_indices]
