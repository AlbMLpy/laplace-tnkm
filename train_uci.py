import argparse
from pathlib import Path
from functools import partial

import jax
jax.config.update("jax_enable_x64", True)

from source.models.LaplaceCPR import LaplaceCPR
from source.general_functions import create_dir_if_not_exists
from source.features import PPFeature, PPNFeature, RBFFeature
from source.data_functions import load_transform_data, load_prepare_data
from source.exp_functions import (
    CVTracker, 
    ll_scorer, 
    nrmse_scorer,
    prepare_train_model,
    get_stats_several_trials, 
)
from configs.uci import ( 
    SCALER,
    CV_NAME,
    ART_DIR, 
    RES_NAME,
    N_TRIALS,
    DATA_DIR,
    PRED_NAME,
    TEST_SIZE,
    TRANSFORM_X,
    TRANSFORM_Y,
    get_exp_config,
)

DATASET_HELP = "Choose dataset:\
    'boston', 'concrete', 'energy', 'kin8nm', 'naval',\
    'power', 'protein', 'wine_red', 'yacht';"
HESS_TYPE_HELP = "Choose hess_type:\
    'full', 'gauss_newton', 'block', 'mf', 'last';"
FMAP_HELP = "Choose feature mapping:\
    'poly', 'poly_norm', 'fourier';"
SCALE_HELP = "Choose float scale hyperparameter for 'fourier' fmap;"
SHIFT_HELP = "Choose positive float shift hyperparameter for fmap;"
SCORER_HELP = "Choose CV scorer function: 'll', 'nrmse';"
TQDM_HELP = "Turn on/off tqdm interactive progress line;"

def argparse_uci():
    parser = argparse.ArgumentParser(description='UCI experiment')
    parser.add_argument('data', type=str, help=DATASET_HELP)
    parser.add_argument('hess', type=str, help=HESS_TYPE_HELP)
    parser.add_argument('fmap', type=str, help=FMAP_HELP)
    parser.add_argument('-fs', '--f_scale', type=float, default=1.0, help=SCALE_HELP)
    parser.add_argument('-fsh', '--f_shift', type=float, default=0.0, help=SHIFT_HELP)
    parser.add_argument('-s', '--scorer', type=str, default='ll', help=SCORER_HELP)
    parser.add_argument('-tqdm', '--tqdm_enable', action="store_true", help=TQDM_HELP)
    return parser.parse_args()

def get_fmap(fmap, fs, shift):
    sh = shift if shift > 0 else None
    if fmap == 'poly': return PPFeature(shift=sh)
    elif fmap == 'poly_norm': return PPNFeature(shift=sh)
    elif fmap == 'fourier': return RBFFeature(l_scale=fs, shift=sh)
    else: raise ValueError()

def get_scorer(scorer):
    if scorer == 'll': return ll_scorer
    elif scorer == 'nrmse': return nrmse_scorer
    else: raise ValueError()

def get_res_dir(args):
    fmap_str = f"{args.fmap}"
    if args.fmap == 'fourier':
        fmap_str += f"_ls{args.f_scale}"
    if args.f_shift > 0:
        fmap_str += f"_sh{args.f_shift}"
    return Path(
        ART_DIR / f'{args.data}/{args.hess}/{args.scorer}/{fmap_str}/training_artifacts/'
    )

if __name__ == '__main__':
    args = argparse_uci()
    data_path = DATA_DIR / f'{args.data}.csv'
    n_samples, d_dim = load_prepare_data(data_path, TEST_SIZE, None)[0].shape
    res_dir = get_res_dir(args)
    create_dir_if_not_exists(res_dir)
    tracker = CVTracker(res_dir, RES_NAME, CV_NAME, PRED_NAME)
    load_data_f = partial(
        load_transform_data, 
        data_path=data_path,
        test_size=TEST_SIZE,
        transform_x=TRANSFORM_X,
        transform_y=TRANSFORM_Y,
        scaler=SCALER,
    )
    train_model_f = prepare_train_model(
        get_exp_config(
            LaplaceCPR,
            n_samples,
            d_dim,
            args.data, 
            args.hess, 
            get_fmap(args.fmap, args.f_scale, args.f_shift),
            get_scorer(args.scorer),
            args.tqdm_enable,
        )
    )
    get_stats_several_trials(
        load_data_f, train_model_f, tracker, N_TRIALS, 
        tqdm_disable=(not args.tqdm_enable)
    )
    tracker.save()
