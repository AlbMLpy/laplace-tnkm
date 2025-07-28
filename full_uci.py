import os
from itertools import product
from joblib import Parallel, delayed

PARALLEL = True
N_JOBS = 12

def run(data_name, hess_type, fmap, shift):
    run_str = f"python train_uci.py {data_name} {hess_type} {fmap} -fsh {shift} -tqdm"
    os.system(run_str)

if __name__ == "__main__":
    data_name_list = ['yacht', 'energy', 'concrete', 'wine_red', 'kin8nm', 'naval', 'power', 'protein', 'boston']
    hess_type_list = ['last', 'block', 'gauss_newton']
    fmap, shift = 'poly_norm', 0.0
    options = list(product(data_name_list, hess_type_list))
    if PARALLEL:
        Parallel(n_jobs=N_JOBS)(
            delayed(run)(data_name, hess_type, fmap, shift) 
            for data_name, hess_type in options
        )
    else:
        for data_name, hess_type in options:
            run(data_name, hess_type, fmap, shift)
