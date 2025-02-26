import numpy as np
import jax.numpy as jnp

# Think about the functionality of this module?

def gd_update(w, dw, lr):
    w -= lr * dw
    return w

def adam_update(w_vec, dw, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    t += 1  # Time step update
    new_m = beta1*m + (1 - beta1)*dw
    new_v = beta2*v + (1 - beta2)*(dw**2)
    # Bias correction
    m_hat = new_m / (1 - beta1**t) 
    v_hat = new_v / (1 - beta2**t)
    # Parameter update
    new_params = w_vec - lr*m_hat/(jnp.sqrt(v_hat) + eps)
    return new_params, new_m, new_v, t

def std_transform(p):
    return jnp.log(jnp.exp(p) + 1)

def sample_w(w_mean, w_std):
    e_noise = jnp.array(np.random.randn(*w_mean.shape))
    return w_mean + e_noise*w_std
