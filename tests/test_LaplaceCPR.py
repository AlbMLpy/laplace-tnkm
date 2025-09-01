import sys
import pytest

import jax
import jax.numpy as jnp
from jax import jit, grad
jax.config.update("jax_enable_x64", True)

import numpy as np
from scipy.stats import linregress

sys.path.append('./')

from source.trackers import SimpleTracker
from source.data_functions import generate_x3_data, scale_data
from source.evaluation import l2_gb_loss, nll, rmse
from source.features import PPNFeature
from source.models.LaplaceCPR import LaplaceCPR

DATA_PARAMS = [(500, 6, 13), (1000, 2, 14), (500, 16, 14)]

@pytest.fixture(params=DATA_PARAMS, scope='module')
def synthetic_train_data(request):
    n_samples, d_dim, seed = request.param
    x_train, y_train, _ = generate_x3_data(n_samples, d_dim, std_err=3, seed=seed)
    return x_train, y_train

@pytest.fixture(scope='module')
def trained_model(synthetic_train_data):
    """ 
    In this setting gamma_w and beta_w are known (n_epoch_vi=1).
    """
    x_train, y_train = synthetic_train_data
    beta_e, gamma_w = 1e-1, 1e-3
    grad_w = jit(grad(l2_gb_loss, argnums=0), static_argnums=(4, 7))
    tracker = SimpleTracker(x_train, y_train, beta_e, gamma_w, loss=l2_gb_loss, grad_w=grad_w)

    rank, m_order, n_epoch = 8, 4, 1000
    model = LaplaceCPR(
        rank=rank, fmap=PPNFeature(), m_order=m_order, n_epoch=n_epoch,
        beta_e=beta_e, gamma_w=gamma_w, pd_mode='lla', hess_type='last',
        hess_th=1e-4, seed=13, n_epoch_vi=1, pd_samples=30,
        pd_sample_seed=14, beta_e_samples=10, tracker=tracker
    )
    model.fit(x_train, y_train)
    return model, tracker.res_dict

class TestLaplaceCPRTraining:
    def test_loss_decrease(self, trained_model):
        _, res = trained_model
        loss_vals = jnp.array(res['loss'])
        assert jnp.all(loss_vals == jnp.sort(loss_vals)[::-1]), "Loss should decrease over training."

    def test_grad_norm_decrease(self, trained_model):
        _, res = trained_model
        gn_vals = jnp.array(res['grad_norm'])
        assert gn_vals[0] > gn_vals[-1], "Gradient norm should decrease over training."

    def _test_grad_trend(self, trained_model):
        _, res = trained_model
        gn_vals = jnp.array(res['grad_norm'])
        slope, _, _, p_value, _ = linregress(np.arange(len(gn_vals)), gn_vals)
        assert slope < 0 and p_value < 0.1, f"Expected downward trend, got slope={slope}."

    def test_grad_moving_avg_decrease(self, trained_model):
        _, res = trained_model
        gn_vals = jnp.array(res['grad_norm'])
        mov_avg = np.convolve(gn_vals, np.ones(10)/10, mode='valid')
        assert mov_avg[0] > mov_avg[-1], "Moving average should decrease."

    def test_local_grad_decrease_ratio(self, trained_model):
        _, res = trained_model
        gn_vals = jnp.array(res['grad_norm'])
        decreases = sum(x > y for x, y in zip(gn_vals, gn_vals[1:]))
        ratio = decreases / (len(gn_vals) - 1)
        assert ratio > 0.5, f"Only {ratio:.1%} of gradient steps decreased (expected > 50%)"


@pytest.fixture(scope='module')
def synthetic_train_test_data():
    d_dim, std_err, data_seed = 6, 3, 13
    x_train, y_train, _ = generate_x3_data(
        500, d_dim, std_err=std_err, seed=data_seed
    )
    x_test, y_test, _ = generate_x3_data(
        100, d_dim, min_v=-5, max_v=5, 
        std_err=std_err, seed=data_seed + 10
    )
    return scale_data(
        x_train, x_test, y_train, y_test, True, False, 'std'
    )

@pytest.fixture(scope='module')
def trained_model_vi(synthetic_train_test_data):
    """ 
    In this setting beta_w is known (n_epoch_vi=1).
    """
    x_train, x_test, y_train, y_test = synthetic_train_test_data
    model = LaplaceCPR(
        rank=4, fmap=PPNFeature(), m_order=8, n_epoch=10,
        beta_e=None, gamma_w=1e-3, pd_mode='lla', hess_type='gauss_newton',
        hess_th=1e-4, seed=13, n_epoch_vi=10, pd_samples=30,
        pd_sample_seed=14, beta_e_samples=10, tracker=None
    )
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test

class _TestLaplaceCPRInference:
    def test_beta_e_precision(self, trained_model_vi):
        model, *t = trained_model_vi
        assert np.abs(model.beta_e - 0.09305423) < 1e-6, "Test beta_e."

    def test_predictions_train(self, trained_model_vi):
        model, x_train, _, y_train, _ = trained_model_vi
        ys_train, ys_std_train = model.predict(x_train, True)
        assert jnp.allclose(
            ys_train[:5], 
            jnp.array(
                [
                    -377.25047459, -372.35587553, -366.40358666, 
                    -363.34453375, -356.68822232
                ]
            )
        ), "Check train predictions."
        assert jnp.allclose(
            ys_std_train[25:30], 
            jnp.array(
                [5.01957394, 4.92824082, 4.66262448, 4.78803918, 4.84689877]
            )
        ), "Check train standard deviations."
        assert np.abs(nll(ys_train, ys_std_train, y_train) - 2.51596445) < 1e-6, "Check train NLL."
        assert np.abs(rmse(ys_train, y_train) - 2.797145237) < 1e-6, "Check train RMSE."

    def test_predictions_test(self, trained_model_vi):
        model, _, x_test, _, y_test = trained_model_vi
        ys_test, ys_std_test = model.predict(x_test, True)
        assert jnp.allclose(
            ys_test[:5], 
            jnp.array(
                [74.2689836, -6.76248461, -58.02371211, -69.34625558, -27.54378316]
            )
        ), "Check test predictions."
        assert jnp.allclose(
            ys_std_test[25:30], 
            jnp.array(
                [
                    75.70363955, 150.49387316, 251.61576873, 
                    390.7567235, 351.54011781
                ]
            )
        ), "Check test standard deviations."
        assert np.abs(nll(ys_test, ys_std_test, y_test) - 30.4807577) < 1e-6, "Check test NLL."
        assert np.abs(rmse(ys_test, y_test) - 210.41505396) < 1e-6, "Check test RMSE."
