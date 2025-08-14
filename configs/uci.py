from pathlib import Path
from typing import Callable

DATA_DIR = Path('./data')
ART_DIR = Path('./artifacts/uci')
N_TRIALS, TEST_SIZE, DATA_SEED = 10, 0.1, 15
TRANSFORM_X, TRANSFORM_Y, SCALER = False, True, 'std'
RES_NAME, CV_NAME, PRED_NAME = 'results', 'cv_train', 'predictions'

RANKS = [2, 4, 8, 16, 24, 32, 48, 64]
M_ORDERS = [2, 4, 8, 16, 24, 32, 48, 64]
HESS_THS = [1e-5, 1e-3, 1e-1]
M_RANKS = [10, 25, 50]
BETA_E, GAMMA_W, N_EPOCH_VI = None, 1e-5, 5
CV_CONFIG = dict(n_splits=3, n_repeats=2, seed=1, n_jobs=1)

def get_exp_config_la_btn(
    model_cls,
    n_samples: int,
    d_dim: int,
    data_name: str, 
    hess_type: str, 
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            fmap=fmap, n_epoch=10, beta_e=BETA_E, gamma_w=GAMMA_W,
            seed=13, pd_mode='lla', hess_type=hess_type, pd_sample_seed=14,
            n_epoch_vi=N_EPOCH_VI,
        ),
        par_flexible=dict(rank=RANKS, m_order=M_ORDERS, hess_th=HESS_THS),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=DATA_DIR / f'{data_name}.csv', n_samples=n_samples,
            d_dim=d_dim, test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )

def get_exp_config_mf_btn(
    model_cls,
    n_samples: int,
    d_dim: int,
    data_name: str, 
    hess_type: str, 
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            fmap=fmap, n_epoch=500, beta_e=BETA_E, gamma_w=GAMMA_W, 
            seed=13, opt_params={'train_mode': 'adam', 'lr': 1e-3},
            n_epoch_vi=N_EPOCH_VI,
        ),
        par_flexible=dict(rank=RANKS, m_order=M_ORDERS),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=DATA_DIR / f'{data_name}.csv', n_samples=n_samples, 
            d_dim=d_dim, test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )

def get_exp_config_sp_btn(
    model_cls,
    n_samples: int,
    d_dim: int,
    data_name: str, 
    hess_type: str, 
    fmap: tuple, 
    scorer: Callable,
    tqdm_enable: bool,
) -> dict:
    return dict(
        model_cls=model_cls,
        scaler=SCALER,
        par_fixed=dict(
            fmap=fmap, n_epoch=500, beta_e=BETA_E, gamma_w=GAMMA_W, 
            seed=13, opt_params={'train_mode': 'adam', 'lr': 1e-3},
            n_epoch_vi=N_EPOCH_VI,
        ),
        par_flexible=dict(rank=RANKS, m_order=M_ORDERS, m_rank=M_RANKS),
        cv_config=CV_CONFIG | dict(scorer=scorer),
        data_config=dict(
            data_path=DATA_DIR / f'{data_name}.csv', n_samples=n_samples, 
            d_dim=d_dim, test_size=TEST_SIZE, seed=DATA_SEED,
        ),
        tqdm_enable=tqdm_enable,
    )
