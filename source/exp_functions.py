import gc
import json
from time import time
from collections import defaultdict
from typing import Optional, Callable

import jax
import jax.numpy as jnp

import numpy as np 
import pandas as pd

from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score, RepeatedKFold

from .optimization import full_grid
from .evaluation import pll, nll, rmse
from .general_functions import update_results_dict, extend_results_dict

def ll_scorer(estimator, X, y):
    """ The higher the better. """
    y_pred, y_std = estimator.predict(X, return_std=True)
    return pll(y_pred, y_std**2, y)

def nrmse_scorer(estimator, X, y):
    """ The higher the better. """
    y_pred = estimator.predict(X, return_std=False)
    return -rmse(y, y_pred)

def model_factory(model_cls, params, scaler: str = 'std'):
    sc = StandardScaler if scaler == 'std' else MinMaxScaler
    return Pipeline([('scaler', sc()), ('model', model_cls(**params))])

def train_model_time(train_model: Callable, x, y):
    start_time = time()
    model, model_info = train_model(x, y)
    elapsed_time = time() - start_time
    model_info['train_time'] = elapsed_time
    return model, model_info

def restrict_options_cpr(options, n_samples, d_dim, seed: Optional[int] = None):
    rs = np.random if seed is None else np.random.RandomState(seed)
    rs.shuffle(options)
    final_options = []
    for option in options:
        rank, m_order = option[0], option[1]
        pp = rank*m_order*d_dim
        if pp <= n_samples and pp <= 1000 and len(final_options) <= 100:
            final_options.append(option)
    return final_options

def prepare_train_model(config: dict) -> Callable:
    cv = RepeatedKFold(
        n_splits=config['cv_config']['n_splits'],
        n_repeats=config['cv_config']['n_repeats'], 
        random_state=config['cv_config']['seed']
    )
    options, option_names = full_grid(config['par_flexible'])
    options = list(options)
    options = restrict_options_cpr(
        options, 
        config['data_config']['n_samples'], 
        config['data_config']['d_dim'], 
        config['data_config']['seed'],
    )

    def train_model(x, y):
        all_scores = []
        for option in tqdm(options, disable=(not config['tqdm_enable'])):
            model_params = dict(zip(option_names, option)) | config['par_fixed']
            model = model_factory(config['model_cls'], model_params, config['scaler'])
            scores = cross_val_score(model, x, y, cv=cv,
                scoring=config['cv_config']['scorer'], 
                n_jobs=config['cv_config']['n_jobs']
            )
            all_scores.append(np.mean(scores))
            
            jax.clear_caches() # To free cache memory
            gc.collect()

        model_info = dict(scores=all_scores, **{k:list(v) for k, v in zip(option_names, list(zip(*options)))})
        best_option = options[np.argmax(all_scores)]
        model_params = dict(zip(option_names, best_option)) | config['par_fixed']
        model = model_factory(config['model_cls'], model_params, config['scaler'])
        model.fit(x, y)  
        return model, model_info
    return train_model

def get_stats_several_trials(
    load_data: Callable,
    train_model: Callable,
    tracker: object,
    n_trials: int = 10,
    tqdm_disable: bool = False,
) -> None:
    for trial in tqdm(range(1, n_trials + 1), disable=tqdm_disable):
        x_train, x_test, y_train, y_test = load_data(split_seed=trial)
        x_train, x_test, y_train, y_test = map(jnp.array, [x_train, x_test, y_train, y_test])
        model, model_info = train_model_time(train_model, x_train, y_train)
        tracker.track(trial, x_train, x_test, y_train, y_test, model, model_info)

class CVTracker:
    def __init__(self, res_dir, res_name, cv_name, pred_name):
        self.res_dict = defaultdict(list)
        self.extra_dict = defaultdict(list)
        self.predictions = None
        self.res_dir = res_dir
        self.res_name = res_name
        self.cv_name = cv_name
        self.pred_name = pred_name

    def track(self, trial, x_train, x_test, y_train, y_test, model, model_info):
        ys_train, ys_std_train = model.predict(x_train, return_std=True)
        ys_test, ys_std_test = model.predict(x_test, return_std=True)

        update_results_dict(self.res_dict, 
            trial=trial,
            rmse_train=rmse(y_train, ys_train),
            rmse_test=rmse(y_test, ys_test),
            nll_train=nll(ys_train, ys_std_train**2, y_train),
            nll_test=nll(ys_test, ys_std_test**2, y_test),
            train_time=model_info.pop('train_time'),
            gamma_w=model['model'].gamma_w,
            beta_e=model['model'].beta_e.item(),
            final_rank=model['model'].rank,
        )
        extend_results_dict(
            self.extra_dict,
            trial=[trial,]*len(model_info['scores']),
            **model_info, 
        )
        if trial == 1:
            self.predictions = dict(y_train=y_train.tolist(), y_test=y_test.tolist(),
                y_train_mean=ys_train.tolist(), y_train_std=ys_std_train.tolist(),
                y_test_mean=ys_test.tolist(), y_test_std=ys_std_test.tolist(),
            )

    def save(self):
        pd.DataFrame(self.res_dict).to_csv(self.res_dir / f'{self.res_name}.csv')
        pd.DataFrame(self.extra_dict).to_csv(self.res_dir / f'{self.cv_name}.csv')
        with open(self.res_dir / f'{self.pred_name}.json', "w") as f:
            json.dump(self.predictions, f)

    def load(self):
        res_df = pd.read_csv(self.res_dir / f'{self.res_name}.csv', index_col=0)
        cv_df = pd.read_csv(self.res_dir / f'{self.cv_name}.csv', index_col=0)
        with open(self.res_dir / f'{self.pred_name}.json', "r") as f:
            pred_dict = json.load(f)
        return res_df, cv_df, pred_dict
