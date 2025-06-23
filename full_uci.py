import os

if __name__ == "__main__":
    data_name_list = ['yacht', 'energy', 'concrete', 'wine_red', 'kin8nm', 'naval', 'power', 'protein', 'boston']
    hess_type_list = ['last', 'block', 'gauss_newton']
    fmap_list = ['poly_norm',]
    scorer_list = ['ll',]

    for data_name in data_name_list:
        for hess_type in hess_type_list:
            for fmap_spec in fmap_list:
                if 'fourier' in fmap_spec:
                    fmap, fs = fmap_spec.split('_')
                    fs = float(fs)
                else:
                    fmap, fs = fmap_spec, -1
                for scorer in scorer_list:
                    run_str = f"python train_uci.py {data_name} {hess_type} {fmap} -fs {fs} -s {scorer} -tqdm"
                    print(run_str)
                    os.system(run_str)
