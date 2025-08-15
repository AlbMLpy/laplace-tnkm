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
from source.general_functions import update_results_dict, create_dir_if_not_exists

sns.set_theme()
warnings.filterwarnings("ignore")

DATA_DIR = Path.cwd().parents[1] / 'data'
RES_DIR = Path('./artifacts/training_artifacts')
MODELS = ['cpr_la', 'cpr_lla']
DATASETS = ['yacht', 'energy', 'concrete', 'wine_red']
TEST_SIZE, SPLIT_SEED = 0.1, 1
TRANSFORM_X, TRANSFORM_Y, SCALER = True, True, 'std'

def train(data_name: str, model_name: str) -> None:
    res_dir = RES_DIR / f'{model_name}/{data_name}/'
    create_dir_if_not_exists(res_dir)
    # Prepare the data:
    x, x_test, y, y_test = load_transform_data(
        DATA_DIR / f'{data_name}.csv', TEST_SIZE, SPLIT_SEED, TRANSFORM_X, TRANSFORM_Y, SCALER)
    x, x_test, y, y_test = map(jnp.array, [x, x_test, y, y_test])
    params = dict(
        m_order=[8,], rank=[10,], fmap=[PPNFeature(),], gamma_w=[1e-6,], beta_e=[1.0,], 
        n_epoch=[100,], pd_mode=[model_name.split('_')[1],], seed=[13,],
        hess_type=['full', 'gauss_newton', 'block', 'mf', 'last'],
        hess_th=[1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2], pd_sample_seed=[1,]
    )
    grad_w = jit(grad(l2_gb_loss, argnums=0), static_argnums=(4, 7))
    grid, param_names = full_grid(params)
    res_dict = defaultdict(list)
    for v in tqdm(grid):
        conf = dict(zip(param_names, v))
        tracker = Tracker(x, y, conf['beta_e'], conf['gamma_w'], l2_gb_loss, grad_w)
        laplace_cpr = LaplaceCPR(tracker=tracker, **conf)
        laplace_cpr.fit(x, y)
        stats = compute_stats_cpr([x, x_test, y, y_test], laplace_cpr, tracker)
        update_results_dict(res_dict, **stats, **conf, **dict(N=x.shape[0], D=x.shape[1]))
    pd.DataFrame(tracker.res_dict).to_csv(res_dir / 'map_tracker.csv')
    pd.DataFrame(res_dict).to_csv(res_dir / 'nll_info.csv')

if __name__ == "__main__":
    for data_name, model_name in product(DATASETS, MODELS):
        print(f"Dataset: {data_name}; Model: {model_name};")
        train(data_name, model_name)
        jax.clear_caches()
        gc.collect()
