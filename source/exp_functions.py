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
from .matrix_operations import ten3tovec
from source.model_functionality import check_zero_cols
from .evaluation import pll, nll, rmse, norm_frob, ecp, wcpi, rce
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

def prepare_train_model(config: dict, logger=None) -> Callable:
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
    n_options = len(options)

    def train_model(x, y):
        all_scores = []
        disable = (not config['tqdm_enable'])
        for idx_opt, option in tqdm(enumerate(options, 1), disable=disable, total=n_options):
            if logger: logger.info(f"##CV option {idx_opt}/{n_options} has started:")
            model_params = dict(zip(option_names, option)) | config['par_fixed']
            model = model_factory(config['model_cls'], model_params, config['scaler'])
            start_time = time()
            scores = cross_val_score(model, x, y, cv=cv,
                scoring=config['cv_config']['scorer'], 
                n_jobs=config['cv_config']['n_jobs']
            )
            elapsed_time = time() - start_time
            if logger: logger.info(f"##Elapsed time={elapsed_time:.3f}")
            all_scores.append(np.mean(scores))
            
            jax.clear_caches() # To free cache memory
            gc.collect()

        model_info = dict(
            scores=all_scores, 
            **{k:list(v) for k, v in zip(option_names, list(zip(*options)))}
        )
        best_option = options[np.argmax(all_scores)]
        model_params = dict(zip(option_names, best_option)) | config['par_fixed']
        model = model_factory(config['model_cls'], model_params, config['scaler'])
        model.fit(x, y)  
        return model, model_info
    return train_model

def prepare_train_model_simple(config: dict, logger=None):
    def train_model(x, y):
        y = y[:, None]
        model_info = dict()
        model = model_factory(config['model_cls'], config['par_fixed'], config['scaler'])
        start_time = time()
        model.fit(x, y)  
        elapsed_time = time() - start_time
        if logger: logger.info(f"##Elapsed time={elapsed_time:.3f}")
        return model, model_info
    return train_model

def get_stats_several_trials(
    load_data: Callable,
    train_model: Callable,
    tracker: object,
    n_trials: int = 10,
    tqdm_disable: bool = False,
    logger = None,
) -> None:
    for trial in tqdm(range(1, n_trials + 1), disable=tqdm_disable):
        if logger: logger.info(f"#Trial {trial}/{n_trials} has started.")
        x_train, x_test, y_train, y_test = load_data(split_seed=trial)
        x_train, x_test, y_train, y_test = map(
            jnp.array, [x_train, x_test, y_train, y_test])
        model, model_info = train_model_time(train_model, x_train, y_train)
        tracker.track(
            trial, x_train, x_test, y_train, y_test, model, model_info)

class CVTracker:
    def __init__(
        self, 
        res_dir, 
        res_name, 
        cv_name, 
        pred_name, 
        seed: Optional[int] = 13
    ):
        self.res_dict = defaultdict(list)
        self.extra_dict = defaultdict(list)
        self.predictions = None
        self.res_dir = res_dir
        self.res_name = res_name
        self.cv_name = cv_name
        self.pred_name = pred_name
        self.alpha = 0.95
        self.n_samples_metric = 300
        self._key = jax.random.PRNGKey(seed)
 
    def track(self, trial, x_train, x_test, y_train, y_test, model, model_info):
        ys_train, ys_std_train = model.predict(x_train, return_std=True)
        ys_test, ys_std_test = model.predict(x_test, return_std=True)
        self._key, subkey = jax.random.split(self._key)
        eps = jax.random.normal(
            subkey, shape=(len(y_test), self.n_samples_metric))
        samples = ys_test[:, None] + eps*ys_std_test[:, None]

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
            ecp_test=ecp(y_test, samples, alpha=self.alpha).item(),
            wcpi_test=wcpi(samples, alpha=self.alpha).item(),
            rce_test=rce(y_test, samples).item(),
        )
        extend_results_dict(
            self.extra_dict,
            trial=[trial,]*len(model_info.get('scores', [])),
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
    
class Tracker:
    def __init__(self, x_train, y_train, beta_e, gamma_w, loss, grad_w):
        self.res_dict = defaultdict(list)
        self.x_train = x_train
        self.y_train = y_train
        self.beta_e = beta_e
        self.gamma_w = gamma_w
        self.loss = loss
        self.grad_w = grad_w

    def track(self, w_ten, kd, fmap):
        w_vec, w_shape = ten3tovec(w_ten), w_ten.shape
        train_loss = self.loss(
            w_vec, kd, self.x_train, self.y_train, 
            fmap, self.gamma_w, self.beta_e, w_shape
        ).item()
        grad_norm = norm_frob(
            self.grad_w(
                w_vec, kd, self.x_train, self.y_train, 
                fmap, self.gamma_w, self.beta_e, w_shape
            )
        ).item()

        update_results_dict(self.res_dict, 
            loss=train_loss,
            grad_norm=grad_norm,
        )

def compute_stats_cpr(data, model, tracker):
    x, x_test, y, y_test = data
    ys_train, _ = model.predict(x, return_std=True)
    ys_test, ys_std_test = model.predict(x_test, return_std=True)
    
    w_hess_evals = np.linalg.eigvals(model.w_hess).real
    
    v_check = (0, int(model.n_epoch // 2), model.n_epoch-1)
    gd_down = [v for i, v in enumerate(tracker.res_dict['grad_norm']) if i in v_check]
    loss_down = [v for i, v in enumerate(tracker.res_dict['loss']) if i in v_check]
    
    res_dict = dict(
        rmse_train=rmse(y, ys_train), rmse_test=rmse(y_test, ys_test), 
        nll_test=nll(ys_test, ys_std_test**2, y_test), loss=tracker.res_dict['loss'][-1], 
        w_norm = norm_frob(model.w_ten).item(), gd_norm=tracker.res_dict['grad_norm'][-1],
        w_new_rank=(~check_zero_cols(model.w_ten).all(axis=0)).sum().item(),
        max_ev=max(w_hess_evals), min_ev=min(w_hess_evals), 
        w_chol_shape=f"{model.w_cholesky.shape}", 
        loss_down=loss_down == sorted(loss_down, reverse=True),
        gd_down=gd_down == sorted(gd_down, reverse=True),
    )
    return res_dict
