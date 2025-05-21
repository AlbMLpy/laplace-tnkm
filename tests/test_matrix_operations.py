import sys
import unittest

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

sys.path.append('./')

from source.matrix_operations import (
    khatri_rao_row, 
    block_diag,
    ten3tovec, 
    vec2ten3
)

class TestFeatures(unittest.TestCase):
    def test_khatri_rao_row(self):
        # prepare the data:
        a = jnp.arange(1, 5).reshape(2, 2)
        b = jnp.arange(1, 7).reshape(2, 3)

        expected = jnp.array(
            [
                [ 1,  2,  3,  2,  4,  6],
                [12, 15, 18, 16, 20, 24],
            ]
        )
        actual = khatri_rao_row(a, b)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_block_diag(self):
        a = jnp.arange(4).reshape(2, 2)
        b = jnp.arange(9).reshape(3, 3)
        expected = jnp.block(
            [
                [a, jnp.zeros((2, 3))],
                [jnp.zeros((3, 2)), b]
            ]
        )
        actual = block_diag([a, b])
        self.assertTrue(jnp.allclose(actual, expected))

    def test_ten3tovec(self):
        w_ten = jnp.array(
            [[[0, 1], [2, 3]], [[3, 4], [4, 5]]]
        )
        expected = jnp.array([0, 2, 1, 3, 3, 4, 4, 5])
        actual = ten3tovec(w_ten)
        self.assertTrue(jnp.allclose(actual, expected))

    def test_vec2ten3(self):
        w_vec = jnp.array([0, 2, 1, 3, 3, 4, 4, 5])
        expected = jnp.array(
            [[[0, 1], [2, 3]], [[3, 4], [4, 5]]]
        )
        actual = vec2ten3(w_vec, *expected.shape)
        self.assertTrue(jnp.allclose(actual, expected))


