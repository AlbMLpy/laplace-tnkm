from pathlib import Path
from typing import Callable

N_TRIALS = 10
TEST_SIZE = 0.1
MODEL_NAME = 'cpr_lla'
DATA_DIR = Path('./data')
ART_DIR = Path('./artifacts/uci')
TRANSFORM_X, TRANSFORM_Y, SCALER = False, True, 'std'
RES_NAME, CV_NAME, PRED_NAME = 'results', 'cv_train', 'predictions'

def get_exp_config(
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
            fmap=fmap,
            n_epoch=10, 
            beta_e=None,
            gamma_w=1e-5, # Decided to fix, otherwise CPD issue;
            pd_mode='lla',
            hess_type=hess_type,
            seed=13,
            n_epoch_vi=5,
            pd_sample_seed=14,
        ),
        par_flexible=dict(
            rank=[2, 4, 8, 16, 24, 32, 48, 64],
            m_order=[2, 4, 8, 16, 24, 32, 48, 64],
            hess_th=[1e-5, 1e-3, 1e-1],
        ),
        cv_config=dict(
            n_splits=3,
            n_repeats=2,
            seed=1,
            scorer=scorer,
            n_jobs=1, # DO NOT CHANGE
        ),
        data_config=dict(
            data_path=DATA_DIR / f'{data_name}.csv',
            n_samples=n_samples,
            d_dim=d_dim,
            test_size=TEST_SIZE,
            seed=15,
        ),
        tqdm_enable=tqdm_enable,
    )
