import gc
import sys
import warnings 
from pathlib import Path
from itertools import product
from collections import defaultdict
sys.path.append(str(Path.cwd().parents[1]))

from tqdm import tqdm

import jax
import jax.numpy as jnp
from jax import jit, grad
jax.config.update("jax_enable_x64", True)

import pandas as pd
import seaborn as sns

from source.features import PPNFeature
from source.evaluation import l2_gb_loss
from source.optimization import full_grid
from source.models.LaplaceCPR import LaplaceCPR
from source.data_functions import load_transform_data
from source.exp_functions import Tracker, compute_stats_cpr
from source.general_functions import (
    update_results_dict, 
    create_dir_if_not_exists,
)
from configs.ablation import (
    RANK,
    SEED,
    MODELS,
    SCALER,
    BETA_E,
    GAMMA_W,
    M_ORDER,
    DATASETS,
    HESS_THS,
    TEST_SIZE,
    ALS_EPOCHS,
    SPLIT_SEED,
    HESS_TYPES,
    TRANSFORM_X,
    TRANSFORM_Y,
    PD_SAMPLE_SEED,
)

sns.set_theme()
warnings.filterwarnings("ignore")

FMAP = PPNFeature()
DATA_DIR = Path.cwd().parents[1] / 'data'
RES_DIR = Path('./artifacts/training_artifacts')

def train(data_name: str, model_name: str) -> None:
    res_dir = RES_DIR / f'{model_name}/{data_name}/'
    create_dir_if_not_exists(res_dir)
    # Prepare the data:
    x, x_test, y, y_test = load_transform_data(
        DATA_DIR / f'{data_name}.csv', 
        TEST_SIZE, SPLIT_SEED, TRANSFORM_X, TRANSFORM_Y, SCALER
    )
    x, x_test, y, y_test = map(jnp.array, [x, x_test, y, y_test])
    params = dict(
        m_order=[M_ORDER,], rank=[RANK,], fmap=[FMAP,], 
        gamma_w=[GAMMA_W,], beta_e=[BETA_E,], n_epoch=[ALS_EPOCHS,], 
        pd_mode=[model_name.split('_')[1],], seed=[SEED,], 
        hess_type=HESS_TYPES, hess_th=HESS_THS, pd_sample_seed=[PD_SAMPLE_SEED,]
    )
    grad_w = jit(grad(l2_gb_loss, argnums=0), static_argnums=(4, 7))
    grid, param_names = full_grid(params)
    res_dict = defaultdict(list)
    for v in tqdm(grid):
        conf = dict(zip(param_names, v))
        tracker = Tracker(
            x, y, conf['beta_e'], conf['gamma_w'], l2_gb_loss, grad_w
        )
        laplace_cpr = LaplaceCPR(tracker=tracker, **conf)
        laplace_cpr.fit(x, y)
        stats = compute_stats_cpr([x, x_test, y, y_test], laplace_cpr, tracker)
        update_results_dict(
            res_dict, **stats, **conf, **dict(N=x.shape[0], D=x.shape[1])
        )
    pd.DataFrame(tracker.res_dict).to_csv(res_dir / 'map_tracker.csv')
    pd.DataFrame(res_dict).to_csv(res_dir / 'nll_info.csv')

if __name__ == "__main__":
    for data_name, model_name in product(DATASETS, MODELS):
        print(f"Dataset: {data_name}; Model: {model_name};")
        train(data_name, model_name)
        jax.clear_caches()
        gc.collect()
