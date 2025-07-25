import os
from itertools import product
from joblib import Parallel, delayed

PARALLEL = True
N_JOBS = 12

def run(data_name, hess_type, fmap, shift):
    run_str = f"python train_uci.py {data_name} {hess_type} {fmap} -fsh {shift} -tqdm"
    print(run_str)
    os.system(run_str)

if __name__ == "__main__":
    data_name_list = ['yacht', 'energy', 'concrete', 'wine_red', 'kin8nm', 'naval', 'power', 'protein', 'boston']
    hess_type_list = ['last', 'block', 'gauss_newton']
    fmap, fmap_shift = 'poly_norm', [0.0, 0.1]
    options = list(product(data_name_list, hess_type_list, fmap_shift))
    if PARALLEL:
        Parallel(n_jobs=N_JOBS)(
            delayed(run)(data_name, hess_type, fmap, shift) 
            for data_name, hess_type, shift in options
        )
    else:
        for data_name, hess_type, shift in options:
            run(data_name, hess_type, fmap, shift)
