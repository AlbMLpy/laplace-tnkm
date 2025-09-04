import sys
import pytest

import jax
import jax.numpy as jnp
from jax import jit, grad
jax.config.update("jax_enable_x64", True)

import numpy as np
from sklearn.datasets import make_friedman2
from sklearn.model_selection import train_test_split

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
    X, y = make_friedman2(n_samples=600, noise=1, random_state=0)
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.16, random_state=42)
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
        rank=10, fmap=PPNFeature(), m_order=40, n_epoch=5,
        beta_e=None, gamma_w=1e-2, pd_mode='lla', hess_type='gauss_newton',
        hess_th=1e-4, seed=13, n_epoch_vi=5, pd_samples=30,
        pd_sample_seed=14, beta_e_samples=10, tracker=None
    )
    model.fit(x_train, y_train)
    return model, x_train, x_test, y_train, y_test

class TestLaplaceCPRInference:
    def test_beta_e_precision(self, trained_model_vi):
        model, *t = trained_model_vi
        assert np.abs(model.beta_e - 0.0012559593826733933) < 1e-6, "Test beta_e."

    def test_predictions_train(self, trained_model_vi):
        model, x_train, _, y_train, _ = trained_model_vi
        ys_train, ys_std_train = model.predict(x_train, True)
        assert jnp.allclose(
            ys_train[:5], 
            jnp.array(
                [
                    255.19251534, 276.18790578, 68.0296712, 
                    985.63203343, 864.4092941,
                ]
            )
        ), "Check train predictions."
        assert jnp.allclose(
            ys_std_train[25:30], 
            jnp.array(
                [52.76310199, 54.01172493, 54.91641517, 48.10949895, 59.84558336]
            )
        ), "Check train standard deviations."
        assert np.abs(nll(ys_train, ys_std_train, y_train) - 3.944720961294799) < 1e-6, "Check train NLL."
        assert np.abs(rmse(ys_train, y_train) - 10.13339733679119) < 1e-6, "Check train RMSE."

    def test_predictions_test(self, trained_model_vi):
        model, _, x_test, _, y_test = trained_model_vi
        ys_test, ys_std_test = model.predict(x_test, True)
        assert jnp.allclose(
            ys_test[:5], 
            jnp.array(
                [1307.79359793, 58.89693868, 647.76475851, 147.0238254, 1047.88973784]
            )
        ), "Check test predictions."
        assert jnp.allclose(
            ys_std_test[25:30], 
            jnp.array(
                [44.17971735, 57.68131017, 160.90904765, 73.31568165, 110.54801714]
            )
        ), "Check test standard deviations."
        assert np.abs(nll(ys_test, ys_std_test, y_test) - 36.452781010913284) < 1e-6, "Check test NLL."
        assert np.abs(rmse(ys_test, y_test) - 143.59825508467918) < 1e-6, "Check test RMSE."
