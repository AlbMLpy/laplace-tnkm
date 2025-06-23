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
SCORER_HELP = "Choose CV scorer function: 'll', 'nrmse';"
TQDM_HELP = "Turn on/off tqdm interactive progress line;"

def argparse_uci():
    parser = argparse.ArgumentParser(description='UCI experiment')
    parser.add_argument('data', type=str, help=DATASET_HELP)
    parser.add_argument('hess', type=str, help=HESS_TYPE_HELP)
    parser.add_argument('fmap', type=str, help=FMAP_HELP)
    parser.add_argument('-fs', '--f_scale', type=float, default=1.0, 
                        help=SCALE_HELP)
    parser.add_argument('-s', '--scorer', type=str, default='ll', 
                        help=SCORER_HELP)
    parser.add_argument('-tqdm', '--tqdm_enable', action="store_true", 
                        help=TQDM_HELP)
    return parser.parse_args()

def get_fmap(fmap, fs=None):
    if fmap == 'poly': return PPFeature()
    elif fmap == 'poly_norm': return PPNFeature()
    elif fmap == 'fourier': return RBFFeature(l_scale=fs)
    else: raise ValueError()

def get_scorer(scorer):
    if scorer == 'll': return ll_scorer
    elif scorer == 'nrmse': return nrmse_scorer
    else: raise ValueError()

if __name__ == '__main__':
    args = argparse_uci()
    data_path = DATA_DIR / f'{args.data}.csv'
    n_samples, d_dim = load_prepare_data(data_path, TEST_SIZE, None)[0].shape
    spec_str = (
        f"{args.fmap}_{args.scorer}" 
        if args.fmap != 'fourier' 
        else f"{args.fmap}{args.f_scale}_{args.scorer}"
    )
    res_dir = Path(
        ART_DIR / f'{args.data}/{args.hess}/{spec_str}/training_artifacts/'
    )
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
            get_fmap(args.fmap, args.f_scale),
            get_scorer(args.scorer),
            args.tqdm_enable,
        )
    )
    get_stats_several_trials(
        load_data_f, train_model_f, tracker, N_TRIALS, 
        tqdm_disable=(not args.tqdm_enable)
    )
    tracker.save()
