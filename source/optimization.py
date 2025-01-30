import numpy as np
import jax.numpy as jnp

# Think about the functionality of this module?

def gd_update(w, dw, lr):
    w -= lr * dw
    return w

def std_transform(p):
    return jnp.log(jnp.exp(p) + 1)

def sample_w(w_mean, w_std):
    e_noise = jnp.array(np.random.randn(*w_mean.shape))
    return w_mean + e_noise*w_std
