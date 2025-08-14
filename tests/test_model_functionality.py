import sys
import unittest
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp
from jax import jit, jacrev, hessian
jax.config.update("jax_enable_x64", True)

sys.path.append('./')

from source.evaluation import l2_gb_loss
from source.matrix_operations import vec2ten3, ten3tovec
from source.features import PPNFeature, pure_poli_features, ppf_q2, prepare_fmap
from source.model_functionality import (
    als_cpd,
    hess_ggn,
    jacob_cpd,
    hess_full,
    hess_diag,
    hess_last,
    init_weights,
    predict_score,
    update_weights,
    hess_block_diag,
    get_fw_hadamard_mtx,
    predict_score_linear,
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
        self.y = jnp.array([1.5, 4.0])
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

    def test_predict_score(self):
        expected = jnp.array([111., 697.])
        actual = predict_score(self.x, self.kd, self.weights, self.fmap)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_predict_score_linear(self):
        expected = 3 * jnp.array([111., 697.])
        actual = predict_score_linear(
            self.x, self.kd, 2*self.weights, self.fmap, self.weights)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_update_weights(self):
        fw_hadamard = get_fw_hadamard_mtx(self.x, self.kd, self.weights, self.fmap)
        expected = jnp.array(
            [[[ 2.36433768e-02,  4.08938675e-02],
                [ 1.49947497e-02,  2.68913284e-02],
                [-2.30250461e-03, -1.11374999e-03]],

            [[ 1.14575093e-01,  2.27235192e-01],
                [ 3.15863569e-01,  6.30582729e-01],
                [ 8.91867290e-01,  1.78950249e+00]]]
        )
        actual = update_weights(
            self.weights.copy(), self.kd, self.weights.shape, 
            self.x, self.y, self.fmap, 1, 1, fw_hadamard,
        )[0]
        self.assertTrue(jnp.allclose(actual, expected))

    def test_als_cpd(self):
        expected = jnp.array(
            [
                0.06898755, 0.04465423, -0.0040124, 0.1381005, 
                0.08938964, -0.00803209, 0.06574435, 0.17968062,  
                0.503937, 0.13160822, 0.35968788, 1.00879012
            ]
        )
        actual = als_cpd(
            ten3tovec(self.weights.copy()), self.kd, 
            self.weights.shape, self.x, self.y, self.fmap, 5, 1, 1
        )
        self.assertTrue(jnp.allclose(actual, expected))

    def test_jacob_cpd(self):
        w_vec_true, w_true_shape = ten3tovec(self.weights), self.weights.shape
        
        expected = jacob_w_jax(self.x, self.kd, w_vec_true, self.fmap, *w_true_shape)
        actual = jacob_cpd(self.weights, self.kd, self.x, self.fmap)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_hess_full(self):
        def hess_full_jax(w_ten, kd, x, y, fmap, gamma_w, beta_e, w_shape):
            hess_f = jit(hessian(l2_gb_loss, argnums=0), static_argnums=(4, 7))
            w_vec = ten3tovec(w_ten)
            hw = hess_f(w_vec, kd, x, y, fmap, gamma_w, beta_e, w_shape)
            return hw
        
        n_samples, d_dim, m_order, rank = 100, 20, 3, 4
        w_ten_test, kd = jnp.array(np.random.randn(d_dim, m_order, rank)), 1
        gamma_w, beta_e = 1.5, 0.5
        fmap, _ = prepare_fmap(PPNFeature(), m_order, False)
        x, y = jnp.array(np.random.randn(n_samples, d_dim)), jnp.array(np.random.randn(n_samples))

        expected = hess_full_jax(w_ten_test, kd, x, y, fmap, gamma_w, beta_e, w_ten_test.shape)
        actual = hess_full(w_ten_test, kd, x, y, fmap, gamma_w, beta_e, mode='full')
        self.assertTrue(jnp.allclose(actual, expected))

    def test_hess_ggn(self):
        expected = jnp.array(
            [
                [3026., 5882.],
                [1446., 2601.]
            ]
        )
        actual = hess_ggn(self.weights, self.kd, self.x, self.fmap, 1, 1)[2:4, 3:5]
        self.assertTrue(jnp.allclose(actual, expected))

    def test_hess_block_diag(self):
        hbd = hess_block_diag(self.weights, self.kd, self.x, self.fmap, 1, 1)
        hf = hess_full(self.weights, self.kd, self.x, self.y, self.fmap, 1, 1, mode='full')
        expected = hf[hbd > 0]
        actual = hbd[hbd > 0]
        self.assertTrue(jnp.allclose(actual, expected))

    def test_hess_diag(self):
        hbd = hess_block_diag(self.weights, self.kd, self.x, self.fmap, 1, 1)
        hd = hess_diag(self.weights, self.kd, self.x, self.fmap, 1, 1)
        expected = hbd[hd > 0]
        actual = hd[hd > 0]
        self.assertTrue(jnp.allclose(actual, expected))

    def test_hess_last(self):
        hbd = hess_block_diag(self.weights, self.kd, self.x, self.fmap, 1, 1)
        hl = hess_last(self.weights, self.kd, self.x, self.fmap, 1, 1)
        _, m_order, rank = self.weights.shape
        expected = hbd[-m_order*rank:, -m_order*rank:]
        actual = hl
        self.assertTrue(jnp.allclose(actual, expected))
