from collections import defaultdict

import numpy as np 

import jax
from jax import jacfwd
import jax.numpy as jnp
import optax
import optax.tree_utils as otu

from source.matrix_operations import vec2ten3, ten3tovec
from source.model_functionality import (
    get_fw_hadamard_mtx,
    predict_score,
    prepare_system,
    check_zero_cols,
    init_weights,
    hess_full,
)
from source.general_functions import update_results_dict, check_nan
from source.evaluation import l2_gb_loss, norm_frob
from source.data_functions import get_batches
from source.optimization import adam_update

### ALS:
def als_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w):
    D, I, R = w_shape
    w_ten = vec2ten3(w_vec, D, I, R)
    fw_hadamard = get_fw_hadamard_mtx(x, kd, w_ten, fmap)
    tracker = defaultdict(list)
    update_results_dict(
        tracker, loss=l2_gb_loss(ten3tovec(w_ten), kd, x, y, fmap, gamma_w, beta_e, w_shape).item(),
        grad_norm=norm_frob(grad_w(ten3tovec(w_ten), kd, x, y, fmap, gamma_w, beta_e, w_shape)).item(),
    )
    for ep in range(n_epoch):
        for ind in range(w_ten.shape[0]):
            # Preprocess:
            k, q = divmod(ind, kd) # q starts from zero -> for fmap
            wk, fk_mtx = w_ten[ind], fmap(x[:, k], q)
            fw_hadamard /= (fk_mtx.dot(wk) + 1e-14) # ZERO DIVISION?
            # Solve linear system:
            alpha = gamma_w / beta_e
            A, b = prepare_system(fk_mtx, fw_hadamard, y)
            A += alpha * jnp.eye(I*R)
            sol = np.linalg.solve(A, b)
            wk = jnp.array(sol.reshape(I, R, order='F')) # Fortran Ordering
            w_ten = w_ten.at[ind].set(wk)
            # Postprocess:
            fw_hadamard *= fk_mtx.dot(wk)
            check_nan(w_ten)
            #_mask = check_zero_cols(w_ten)
            #mask = ~_mask.all(axis=0)
        update_results_dict(
            tracker, loss=l2_gb_loss(ten3tovec(w_ten), kd, x, y, fmap, gamma_w, beta_e, w_shape).item(),
            grad_norm=norm_frob(grad_w(ten3tovec(w_ten), kd, x, y, fmap, gamma_w, beta_e, w_shape)).item(),
        )
    return ten3tovec(w_ten), tracker

### Adam:
def adam_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w, lr, bs):
    m, v, t = jnp.zeros_like(w_vec), jnp.zeros_like(w_vec), 0
    tracker = defaultdict(list)
    update_results_dict(
        tracker, loss=l2_gb_loss(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape).item(),
        grad_norm=norm_frob(grad_w(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape)).item(),
    )
    for ep in range(n_epoch):
        for x_batch, y_batch in get_batches(x, y, batch_size=bs):
            dw = grad_w(w_vec, kd, x_batch, y_batch, fmap, gamma_w, beta_e, w_shape)
            w_vec, m, v, t = adam_update(w_vec, dw, m, v, t, lr)
        update_results_dict(
            tracker, loss=l2_gb_loss(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape).item(),
            grad_norm=norm_frob(grad_w(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape)).item(),
        )
    return w_vec, tracker

### L-BFGS Jax:
def prep_loss(kd, w_shape, x, y, fmap, gamma_w, beta_e):
    def loss_fn(w_vec):
        return l2_gb_loss(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape)
    return loss_fn

def lbfgs_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w, tol):
    fun = prep_loss(kd, w_shape, x, y, fmap, gamma_w, beta_e)
    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun
        )
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < n_epoch) & (err >= tol))

    init_carry = (w_vec, opt.init(w_vec))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state

### Gauss-Newton (LM):
def gauss_newton_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w, lm_mode=False, lambda_init=1e-3):
    def residual(w_vec):
        return jnp.sqrt(beta_e)*(y - predict_score(x, kd, vec2ten3(w_vec, *w_shape), fmap))
    loss_fn = prep_loss(kd, w_shape, x, y, fmap, gamma_w, beta_e)
    tracker = defaultdict(list)
    update_results_dict(
        tracker, loss=loss_fn(w_vec).item(),
        grad_norm=norm_frob(grad_w(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape)).item(),
    )
    lambda_damping = lambda_init if lm_mode else 0.0  # Initialize damping factor
    for ep in range(n_epoch):
        J = jacfwd(residual)(w_vec) # Compute Jacobian J(w)
        A = J.T.dot(J) + (lambda_damping + gamma_w) * jnp.eye(J.shape[1]) # Regularized Hessian approx
        b = J.T.dot(residual(w_vec)) # Gradient approximation
        delta_w = np.linalg.solve(A, -b)
        new_w_vec = w_vec + delta_w 
        if lm_mode:
            new_loss = loss_fn(new_w_vec).item()
            old_loss = tracker['loss'][-1]
            # Adapt lambda: Reduce if loss improves, increase if it worsens
            if new_loss < old_loss:
                lambda_damping *= 0.5 # Reduce damping (trusts Gauss-Newton more)
                w_vec = new_w_vec  # Accept the step
            else:
                lambda_damping *= 2.0  # Increase damping (fallback to gradient descent)
                continue
        else:
            w_vec = new_w_vec
        update_results_dict(
            tracker, loss=loss_fn(w_vec).item(),
            grad_norm=norm_frob(grad_w(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape)).item(),
        )
    return w_vec, tracker

def cpd_solution(opt_name, w_vec, kd, w_shape, x, y, fmap, gamma_w, beta_e, grad_w_jax):
    if opt_name == 'ALS':
        n_epoch = 100
        w_vec, tracker = als_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w_jax)
    elif opt_name == 'Adam':
        n_epoch, lr, bs = 200, 3e-2, 25
        w_vec, tracker = adam_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w_jax, lr, bs)
    elif opt_name == 'LBFGS':
        n_epoch, tol = 200, 1e-6
        w_vec, _ = lbfgs_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w_jax, tol)
    elif opt_name == 'GN':
        n_epoch, lm_mode = 200, False
        w_vec, tracker = gauss_newton_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w_jax, lm_mode)
    elif opt_name == 'LM':
        n_epoch, lm_mode, lambda_init = 200, True, 1e-3
        w_vec, tracker = gauss_newton_cpd(w_vec, kd, w_shape, x, y, fmap, n_epoch, gamma_w, beta_e, grad_w_jax, lm_mode, lambda_init)
    return w_vec
