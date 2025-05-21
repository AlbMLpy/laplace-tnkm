import sys
import unittest
from functools import partial

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np

sys.path.append('./')

from source.model_functionality import (
    hess_full,
    init_weights, 
    get_fw_hadamard_mtx,
    get_ww_hadamard_mtx,
    get_updated_als_factor,
)
from source.features import pure_poli_features, ppf_q2, PPNFeature, prepare_fmap
from source.optimization import hess_full_jax


class TestModelFunctionality(unittest.TestCase):
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

    def test_hess_full(self):
        n_samples, d_dim, m_order, rank = 100, 20, 3, 4
        w_ten_test, kd = jnp.array(np.random.randn(d_dim, m_order, rank)), 1
        gamma_w, beta_e = 1.5, 0.5
        fmap, _ = prepare_fmap(PPNFeature(), m_order, False)
        x, y = jnp.array(np.random.randn(n_samples, d_dim)), jnp.array(np.random.randn(n_samples))

        expected = hess_full_jax(w_ten_test, kd, x, y, fmap, gamma_w, beta_e, w_ten_test.shape)
        actual = hess_full(w_ten_test, kd, x, y, fmap, gamma_w, beta_e, mode='full')
        self.assertTrue(jnp.allclose(actual, expected)) #norm_frob(H_jax - H_hand)
