from typing import Callable

import optax
from flax import nnx

from source.models.LaplaceBNN import SimpleNN

FMAP, FSHIFT = 'poly_norm', 0.0
MODELS = ['la_btn', 'mf_btn', 'sp_btn']
HESS_TYPES = ['last', 'block', 'gauss_newton']
DATASETS = [
    'yacht', 'energy', 'concrete', 'wine_red', 
    'kin8nm', 'naval', 'power', 'protein', 'boston',
]

N_TRIALS = 10
DATA_SEED = 15
TEST_SIZE = 0.1
TRANSFORM_X, TRANSFORM_Y, SCALER = False, True, 'std'
RES_NAME, CV_NAME, PRED_NAME = 'results', 'cv_train', 'predictions'

N_EPOCH_VI = 5
GD_EPOCHS = 100
ALS_EPOCHS = 10
BETA_E, GAMMA_W = None, 1e-5
HESS_THS = [1e-5, 1e-3, 1e-1]
RANKS = [2, 4, 8, 16, 24, 32, 48, 64]
M_ORDERS = [2, 4, 8, 16, 24, 32, 48, 64]
CV_CONFIG = dict(n_splits=3, n_repeats=2, seed=1, n_jobs=1)

def get_exp_config_la_btn(
    model_cls,
    n_samples: int,
    d_dim: int,
    data_path: str, 
    hess_type: str, 
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            fmap=fmap, n_epoch=ALS_EPOCHS, beta_e=BETA_E, gamma_w=GAMMA_W,
            seed=13, pd_mode='lla', hess_type=hess_type, pd_sample_seed=14,
            n_epoch_vi=N_EPOCH_VI,
        ),
        par_flexible=dict(rank=RANKS, m_order=M_ORDERS, hess_th=HESS_THS),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=data_path, n_samples=n_samples,
            d_dim=d_dim, test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )

def get_exp_config_mf_btn(
    model_cls,
    n_samples: int,
    d_dim: int,
    data_path: str, 
    hess_type: str, 
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            fmap=fmap, n_epoch=GD_EPOCHS, beta_e=BETA_E, gamma_w=GAMMA_W, 
            seed=13, opt_params={'train_mode': 'adam', 'lr': 2e-3},
            n_epoch_vi=N_EPOCH_VI,
        ),
        par_flexible=dict(rank=RANKS, m_order=M_ORDERS),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=data_path, n_samples=n_samples, 
            d_dim=d_dim, test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )

def get_exp_config_sp_btn(
    model_cls,
    n_samples: int,
    d_dim: int,
    data_path: str, 
    hess_type: str, 
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            fmap=fmap, n_epoch=GD_EPOCHS, beta_e=BETA_E, gamma_w=GAMMA_W, 
            seed=13, opt_params={'train_mode': 'adam', 'lr': 2e-3},
            n_epoch_vi=N_EPOCH_VI, m_rank=16,
        ),
        par_flexible=dict(rank=RANKS, m_order=M_ORDERS),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=data_path, n_samples=n_samples, 
            d_dim=d_dim, test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )

def get_exp_config_la_bnn(
    model_cls,
    n_samples: int,
    d_dim: int,
    data_path: str, 
    hess_type: str, 
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            nn_arch=SimpleNN(d_dim, 100, 1, nnx.Rngs(0)),
            opt=optax.adam(1e-3),
            batch_size=100,
            n_epoch=500,
            beta_e=BETA_E,
            gamma_w=1e-1,
            seed=42,
            curv='full',
            linearized=True,
            shuffle=True,
            verbose=False,
        ),
        par_flexible=dict(),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=data_path, n_samples=n_samples, 
            d_dim=d_dim, test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )
