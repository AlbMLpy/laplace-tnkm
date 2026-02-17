import sys
import argparse
from time import time
from pathlib import Path
from functools import partial
from collections import defaultdict
from typing import Optional, Callable
sys.path.append(str(Path.cwd().parents[1]))

import jax
from jax import Array
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
import pandas as pd
from tqdm import tqdm

from source.evaluation import rmse, nll
from source.models.LaplaceCPR import LaplaceCPR
from source.features import RBFFeature, rbf_features
from source.model_functionality import predict_score, als_cpd_reg_w
from source.data_functions import load_prepare_data, scale_data
from source.general_functions import (
    update_results_dict,
    create_dir_if_not_exists, 
)
from configs.airline import Config


DATA_DIR = Path.cwd().parents[1] / 'data'
ART_DIR = Path('./artifacts/training_artifacts')

RANK_HELP = "Choose LA-TNKM model rank:\
    5, 10, 15, 20 (generally any value > 1);"
N_SAMPLE_HELP = "Choose the number of samples \
    used for Airline data: 10000, 100000, 1000000, 5929413;"
TQDM_HELP = "Turn on/off tqdm interactive progress line;"

def argparse_airline():
    parser = argparse.ArgumentParser(description='Airline experiment')
    parser.add_argument('rank', type=int, help=RANK_HELP)
    parser.add_argument('n_sample', type=str, help=N_SAMPLE_HELP)
    parser.add_argument(
        '-tqdm', '--tqdm_enable', action="store_true", help=TQDM_HELP)
    return parser.parse_args()

def get_res_dir(args, dir_path=ART_DIR):
    return Path(dir_path / f'n_sample_{args.n_sample}/rank_{args.rank}/')

def load_transform_data(
    data_path, 
    test_size, 
    split_seed, 
    transform_x: bool = True, 
    transform_y: bool = True, 
    scaler: str = 'std',
    n_sample: Optional[int] = None, 
) -> tuple[Array]:
    x, x_test, y, y_test = load_prepare_data(
        data_path, test_size, split_seed, n_sample)
    x, x_test, y, y_test = scale_data(
        x, x_test, y, y_test, transform_x, transform_y, scaler)
    return map(jnp.array, [x, x_test, y, y_test])

def get_stats_several_trials(
    config,
    load_data: Callable,
    tracker: object,
    tqdm_disable: bool = False,
):
    for trial in tqdm(range(config.n_trials), disable=tqdm_disable):
        model_seed = data_seed = config.seed + trial
        x, x_test, y, y_test = load_data(split_seed=data_seed)
        LSCALE = x.std(axis=0).mean().item()
        # Train T-KRR:
        fmap = partial(rbf_features, order=config.m_order, lscale=LSCALE)
        st_time = time()
        w_tkrr, _ = als_cpd_reg_w(
            x, y, config.m_order, fmap, config.rank, config.init_type, 
            config.n_epoch_tkrr, config.alpha, model_seed,
        )
        w_tkrr.block_until_ready()
        time_tkrr = time() - st_time
        # Train LA-TNKM using T-KRR as init:
        model = LaplaceCPR(
            config.rank, RBFFeature(l_scale=LSCALE), config.m_order, 
            config.n_epoch_la_tnkm, config.beta_e, config.gamma_w,  
            seed=model_seed, n_epoch_vi=config.n_epoch_vi, 
            pd_samples=config.pd_samples, pd_sample_seed=model_seed + 1, 
            beta_e_samples=config.beta_e_samples,
        )
        st_time = time()
        model.fit(x, y, w_ten=w_tkrr.copy())
        time_la_tnkm = time_tkrr + time() - st_time 
        # Save the results:
        model_info = dict(
            time_tkrr=time_tkrr, time_la_tnkm=time_la_tnkm, 
            w_tkrr=w_tkrr, fmap=fmap
        )
        tracker.track(trial, x, x_test, y, y_test, model, model_info)

class Tracker:
    def __init__(self, res_dir, res_name):
        self.res_dict = defaultdict(list)
        self.res_dir = res_dir
        self.res_name = res_name
 
    def track(self, trial, x, x_test, y, y_test, model, model_info):
        ys_test, ys_std_test = model.predict(x_test, return_std=True)
        ys_test_tkrr = predict_score(
            x_test, 1, model_info['w_tkrr'], model_info['fmap'])
        
        update_results_dict(self.res_dict, trial=trial,
                            mse_test=rmse(y_test, ys_test)**2,
                            mse_test_tkrr=rmse(y_test, ys_test_tkrr)**2,
                            nll_test=nll(ys_test, ys_std_test**2, y_test),
                            train_time_tkrr=model_info.pop('time_tkrr'),
                            train_time_la_tnkm=model_info.pop('time_la_tnkm'))

    def save(self):
        pd.DataFrame(
            self.res_dict).to_csv(self.res_dir / f'{self.res_name}.csv')

    def load(self):
        return pd.read_csv(
            self.res_dir / f'{self.res_name}.csv', index_col=0)
    
if __name__ == '__main__':
    # Get parameters and prepare dir:
    args = argparse_airline()
    n_sample, rank = int(args.n_sample), int(args.rank)
    gamma_w = 100/n_sample
    if n_sample > 100000: gamma_w = 0.1
    res_dir = get_res_dir(args)
    create_dir_if_not_exists(res_dir)
    # Prepare config:
    config = Config(rank=rank, gamma_w=gamma_w, alpha=100/n_sample)
    # Define metrics tracker:
    tracker = Tracker(res_dir, config.res_name)
    # Define data loader:
    test_size = int(np.floor(n_sample/3))
    load_data_f = partial(
        load_transform_data, data_path=DATA_DIR / f'airline.csv', 
        test_size=test_size, transform_x=config.transform_x, 
        transform_y=config.transform_y, scaler=config.scaler, 
        n_sample=n_sample,
    )
    # Start experiment:
    get_stats_several_trials(
        config, load_data_f, tracker, not args.tqdm_enable)
    tracker.save()
