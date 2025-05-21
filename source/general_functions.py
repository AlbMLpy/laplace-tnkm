import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pandas as pd

def query_df(df_res: pd.DataFrame, query: dict) -> pd.DataFrame:
    mask = 1
    for col, val in query.items():
       mask &= (df_res[col] == val)
    return df_res[mask]

def extend_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].extend(value.copy())

def update_results_dict(res: dict, **kwargs) -> None:
    for key, value in kwargs.items():
        res[key].append(value)

def create_dir_if_not_exists(directory: str) -> os.PathLike:
    path = Path(directory)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    return path

def prepare_for_dump(dt: dict) -> dict:
    for k in dt.keys():
        if isinstance(dt[k][0], np.ndarray):
            dt[k] = [list(v) for v in dt[k]]
    return dt

def check_nan(w_ten):
    if jnp.any(jnp.isnan(w_ten)) or (w_ten.max().item() > 1e30):
        raise ValueError("NaNs / big numbers detected in model parameters! Stop!")
