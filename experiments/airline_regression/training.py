import os
import sys
import argparse
from pathlib import Path
from itertools import product
from joblib import Parallel, delayed
sys.path.append(str(Path.cwd().parents[1]))

from configs.airline import RANKS, N_SAMPLES

PARALLEL_HELP = "Turn on/off parallel mode;"
N_JOBS_HELP = "How many processes to use for computations;"

def argparse_airline():
    parser = argparse.ArgumentParser(description='Airline experiment')
    parser.add_argument('-p', '--parallel', action='store_true', help=PARALLEL_HELP)
    parser.add_argument('-n', '--n_jobs', type=int, default=12, help=N_JOBS_HELP)
    return parser.parse_args()

def run(rank, n_sample):
    run_str = f"python train_part.py {rank} {n_sample} -tqdm"
    print(run_str)
    os.system(run_str)

if __name__ == "__main__":
    args = argparse_airline()
    options = [v for v in product(RANKS, N_SAMPLES)]
    if args.parallel:
        Parallel(n_jobs=args.n_jobs)(
            delayed(run)(rank, n_sample) for rank, n_sample in options
        )
    else:
        for rank, n_sample in options: run(rank, n_sample)
