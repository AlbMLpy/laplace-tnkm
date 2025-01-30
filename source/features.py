from typing import Callable
from functools import partial
from collections import namedtuple

import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike

Feature = tuple
PPFeature = namedtuple('PPFeature', 'name', defaults=['ppf'])
FFeature = namedtuple('FFeature', 'p_scale, name', defaults=[1, 'ff'])
RBFFeature = namedtuple('RBFFeature', 'l_scale, name', defaults=[1, 'rbff'])
FeatureMap = Callable[..., Array]

@partial(jit, static_argnums=2)
def pure_poli_features(x: ArrayLike, q: int, order: int) -> Array:
    """ 
    Pure polinomial features matrix for x. 

    References: "Quantized Fourier and Polynomial Features for more 
        Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Definition 3.1).
    """
    return jnp.power(x[:, None], jnp.arange(order))

@jit
def ppf_q2(x: ArrayLike, q: int) -> Array:
    """ 
    Quantized pure polinomial features matrix for x. 
    
    References: "Quantized Fourier and Polynomial Features for more 
        Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Definition 3.4).

    NOTE: q should start with 0 -> [log2(m_order) - 1] including
    """
    return jnp.power(x[:, None], jnp.array([0, 2**q]))

@partial(jit, static_argnums=[2, 3, 4])
def gaussian_kernel_features(
    x: ArrayLike,
    q: int,  # Dummy
    order: int, 
    lscale: float = 1, 
    domain_bound: float = 1,
) -> Array:
    """ 
    Gaussian (squared exp.) kernel features matrix for x. 

    References: "Hilbert Space Methods for Reduced-Rank Gaussian Process Regression", 
        Simo Särkkä, (formulas 56, 68(d=1, s=1)).
    """
    x = (x + domain_bound)
    w_scaled = jnp.pi * jnp.arange(1, order + 1) / (2 * domain_bound)
    sd = jnp.sqrt(2 * jnp.pi) * lscale * jnp.exp(-jnp.power(lscale * w_scaled, 2) / 2)
    return jnp.sqrt(sd / domain_bound) * jnp.sin(jnp.outer(x, w_scaled)) 

@partial(jit, static_argnums=[2,])
def fourier_features(x: ArrayLike, q: int, m_order: int, p_scale: float = 1) -> Array:
    """ 
    Fourier Features matrix for x. 

    References: 
        - "Learning multidimensional Fourier series with tensor trains",
            Sander Wahls, Visa Koivunen, H Vincent Poor, Michel Verhaegen.
        - "Quantized Fourier and Polynomial Features for more 
            Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Definition 3.2).
    """
    w = jnp.arange(-m_order / 2, m_order / 2)
    return jnp.exp(1j * 2 * jnp.pi * jnp.outer(x, w) / p_scale)

@jit
def ff_q2(
    x: ArrayLike, 
    q: int, 
    m_order: int, 
    k_d: int, 
    p_scale: float = 1
) -> Array:
    """ 
    Quantized Fourier Features matrix for x. 

    References: 
        - "Learning multidimensional Fourier series with tensor trains",
            Sander Wahls, Visa Koivunen, H Vincent Poor, Michel Verhaegen.
        - "Quantized Fourier and Polynomial Features for more 
            Expressive Tensor Network Models", Frederiek Wesel, Kim Batselier, (Corollary 3.6).

    NOTE: q should start with 0 -> [log2(m_order) - 1] including
    """
    return  jnp.hstack(
        (
            jnp.exp(-1j * jnp.pi * x * m_order / (k_d * p_scale))[:, None], 
            jnp.exp(1j * jnp.pi * (-x * m_order / k_d + 2*x*(2**(q))) / p_scale)[:, None]
        ),
    )
