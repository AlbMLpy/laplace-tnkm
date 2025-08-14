import os
import argparse
from itertools import product, chain
from joblib import Parallel, delayed

N_JOBS = 12
PARALLEL = True

FMAP, FSHIFT = 'poly_norm', 0.0
MODELS = ['la_btn', 'mf_btn', 'sp_btn']
DATASETS = ['yacht', 'energy', 'concrete', 'wine_red', 'kin8nm', 'naval', 'power', 'protein', 'boston']
HESS_TYPES = ['last', 'block', 'gauss_newton']

MODEL_HELP = "Choose model: 'all', 'la_btn', 'mf_btn', 'sp_btn';"

def argparse_uci():
    parser = argparse.ArgumentParser(description='UCI experiment')
    parser.add_argument('model', type=str, help=MODEL_HELP)
    return parser.parse_args()

def get_options(model, datasets, hess_types):
    if model == 'la_btn':
        return [(model, *v) for v in product(datasets, hess_types)]
    elif model in MODELS:
        return [(model, data, 'mf') for data in datasets]
    else:
        raise ValueError(f"Bad model name: {model}")

def run(model_name, data_name, hess_type, fmap, shift):
    run_str = f"python train_uci.py {model_name} {data_name} {hess_type} {fmap} -fsh {shift} -tqdm"
    os.system(run_str)

if __name__ == "__main__":
    args = argparse_uci()
    if args.model == 'all':
        options = [get_options(model, DATASETS, HESS_TYPES) for model in MODELS]
        options = list(chain.from_iterable(options))
    else:
        options = get_options(args.model, DATASETS, HESS_TYPES)
    
    if PARALLEL:
        Parallel(n_jobs=N_JOBS)(
            delayed(run)(model_name, data_name, hess_type, FMAP, FSHIFT) 
            for model_name, data_name, hess_type in options
        )
    else:
        for model_name, data_name, hess_type in options:
            run(model_name, data_name, hess_type, FMAP, FSHIFT)
