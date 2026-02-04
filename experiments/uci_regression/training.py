import os
import sys
import argparse
from pathlib import Path
from itertools import product, chain
from joblib import Parallel, delayed
sys.path.append(str(Path.cwd().parents[1]))

from configs.uci import ( 
    FMAP,
    FSHIFT,
    MODELS,
    DATASETS,
    HESS_TYPES,
)

MODEL_HELP = "Choose model: 'all', 'la_btn', 'mf_btn', 'sp_btn', 'la_bnn';"
PARALLEL_HELP = "Turn on/off parallel mode;"
N_JOBS_HELP = "How many processes to use for computations;"

def argparse_uci():
    parser = argparse.ArgumentParser(description='UCI experiment')
    parser.add_argument('model', type=str, help=MODEL_HELP)
    parser.add_argument('-p', '--parallel', action='store_true', help=PARALLEL_HELP)
    parser.add_argument('-n', '--n_jobs', type=int, default=12, help=N_JOBS_HELP)
    return parser.parse_args()

def get_options(model, datasets, hess_types):
    if model == 'la_btn':
        return [(model, *v) for v in product(datasets, hess_types)]
    elif model in MODELS:
        hess = 'full' if model == 'la_bnn' else 'mf'
        return [(model, data, hess) for data in datasets]
    else:
        raise ValueError(f"Bad model name: {model}")

def run(model_name, data_name, hess_type, fmap, shift):
    if model_name == 'la_bnn': fmap = 'alt'
    run_str = f"python train_part.py {model_name} {data_name} {hess_type} {fmap} -fsh {shift} -tqdm"
    os.system(run_str)

if __name__ == "__main__":
    args = argparse_uci()
    if args.model == 'all':
        options = [get_options(model, DATASETS, HESS_TYPES) for model in MODELS]
        options = list(chain.from_iterable(options))
    else:
        options = get_options(args.model, DATASETS, HESS_TYPES)
    
    if args.parallel:
        Parallel(n_jobs=args.n_jobs)(
            delayed(run)(model_name, data_name, hess_type, FMAP, FSHIFT) 
            for model_name, data_name, hess_type in options
        )
    else:
        for model_name, data_name, hess_type in options:
            run(model_name, data_name, hess_type, FMAP, FSHIFT)
