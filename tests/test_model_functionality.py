import sys
import unittest
from functools import partial

import jax
import jax.numpy as jnp
from jax import jit, jacrev, hessian
jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.append('./')

from source.optimization import hess_full_jax
from source.matrix_operations import vec2ten3, ten3tovec
from source.features import pure_poli_features, ppf_q2, PPNFeature, prepare_fmap
from source.model_functionality import (
    jacob_cpd,
    hess_full,
    init_weights,
    predict_score,
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    predict_score_linear,
    get_updated_als_factor,
)

def predict_score_vec(x, kd, w_vec, fmap, D, I, R):
    w_ten = vec2ten3(w_vec, D, I, R)
    return predict_score(x, kd, w_ten, fmap)

jacob_w_jax = jit(jacrev(predict_score_vec, argnums=2), static_argnums=(3, 4, 5, 6))

class TestModelFunctionality(unittest.TestCase):
    def setUp(self):
        self.x = jnp.array(
            [
                [1.0, 2],
                [2, 3],
            ]
        )
        self.kd = 1
        self.weights = jnp.array(
            [
                [[1.0, 0], [2, 1], [3, 2]], 
                [[0, 1], [1, 2], [2, 3]]
            ]
        )
        self.fmap = partial(pure_poli_features, order=3) 

    def test_init_weights_non_quant(self):
        m_order, rank, d_dim, q_base = 13, 5, 4, None
        temp, _ = init_weights(m_order, rank, d_dim, q_base)
        
        expected = jnp.array([d_dim, m_order, rank])
        actual = jnp.array(temp.shape)
        self.assertTrue(jnp.allclose(actual, expected))
    
    def test_init_weights_quant(self):
        m_order, rank, d_dim, q_base = 16, 5, 4, 2
        temp, _ = init_weights(m_order, rank, d_dim, q_base)
        
        expected = jnp.array(
            [d_dim*int(jnp.log2(m_order)), q_base, rank])
        actual = jnp.array(temp.shape)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_init_weights_bad_m_order(self):
        # Quantized setting:
        m_order, rank, d_dim, q_base = 13, 5, 4, 2
        with self.assertRaises(ValueError):
            init_weights(m_order, rank, d_dim, q_base)

    def test_get_updated_als_factor_ww(self):
        n, f_dim = 3, 2
        fk_mtx = jnp.ones((n, f_dim))
        fw_hadamard = jnp.array(
             [[1.0, 2], [2, 4], [4, 8]]
        )
        ww_hadamard = jnp.array([[1.0, 3], [3, 5]])
        y = jnp.array([1.0, 0, 1])
        alpha = 1.0
        
        expected = jnp.array(
            [
                [0.03846154, 0.03846154],
                [0.03846154, 0.03846154]
            ]
        )
        actual = get_updated_als_factor(
            fk_mtx, 
            fw_hadamard,
            ww_hadamard, 
            y, 
            alpha,
            pinv=False,
            ww_reg=True,
        )
        self.assertTrue(jnp.allclose(actual, expected))

    def test_get_fw_hadamard_mtx_quant(self):
        x = jnp.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ]
        )
        k_d = 2
        weights = jnp.array(
            [
                [[1, 2], [2, 3]], 
                [[0, 1], [1, 0]],
                [[1, 2], [2, 3]], 
                [[0, 1], [1, 0]]
            ]
        )
        feature_map = ppf_q2

        expected = jnp.array(
            [
                [9., 25],
                [400., 64.],
                [3969., 121.],
                [20736., 196.]
            ]
        )
        actual = get_fw_hadamard_mtx(x, k_d, weights, feature_map)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_get_fw_hadamard_mtx_non_quant(self):
        x = jnp.array(
            [
                [1, 1],
                [2, 2],
                [3, 3],
                [4, 4],
            ]
        )
        k_d = 1
        weights = jnp.array(
            [
                [[1, 2], [2, 3], [3, 4]], 
                [[0, 1], [1, 0], [1, 1]]
            ]
        )
        feature_map = partial(pure_poli_features, order=3)

        expected = jnp.array(
            [
                [  12.,   18.],
                [ 102.,  120.],
                [ 408.,  470.],
                [1140., 1326.]
            ]
        )
        actual = get_fw_hadamard_mtx(x, k_d, weights, feature_map)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_get_ww_hadamard_mtx(self):
        weights = jnp.array(
            [
                [[1, 2], [2, 3], [3, 4]], 
                [[0, 1], [1, 0], [1, 1]]
            ]
        )

        expected = jnp.array([[28., 20.], [20., 58.]])
        actual = get_ww_hadamard_mtx(weights)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_predict_score(self):
        expected = jnp.array([111., 697.])
        actual = predict_score(self.x, self.kd, self.weights, self.fmap)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_jacob_cpd(self):
        w_vec_true, w_true_shape = ten3tovec(self.weights), self.weights.shape
        
        expected = jacob_w_jax(self.x, self.kd, w_vec_true, self.fmap, *w_true_shape)
        actual = jacob_cpd(self.weights, self.kd, self.x, self.fmap)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_hess_full(self):
        n_samples, d_dim, m_order, rank = 100, 20, 3, 4
        w_ten_test, kd = jnp.array(np.random.randn(d_dim, m_order, rank)), 1
        gamma_w, beta_e = 1.5, 0.5
        fmap, _ = prepare_fmap(PPNFeature(), m_order, False)
        x, y = jnp.array(np.random.randn(n_samples, d_dim)), jnp.array(np.random.randn(n_samples))

        expected = hess_full_jax(w_ten_test, kd, x, y, fmap, gamma_w, beta_e, w_ten_test.shape)
        actual = hess_full(w_ten_test, kd, x, y, fmap, gamma_w, beta_e, mode='full')
        self.assertTrue(jnp.allclose(actual, expected))
        