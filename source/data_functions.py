from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def load_prepare_data(
    data_path: str, 
    test_size: float, 
    split_seed: int, 
    n_sample: Optional[int] = None,
) -> tuple[pd.DataFrame]:
    df = pd.read_csv(data_path, header=0)
    df = df.sample(n_sample, random_state=split_seed) if n_sample else df
    target_col = 'target' if 'target' in df.columns else 'class'
    return train_test_split(
        df[[v for v in df.columns if v != target_col]], 
        df[target_col], 
        test_size=test_size,
        random_state=split_seed,
    )

def get_df_stats(data_path) -> tuple[int, int]:
    _x1, _x2, _, _ = load_prepare_data(data_path, 0.2, split_seed=None)
    n_samples, d_dim = _x1.shape[0] + _x2.shape[0], _x1.shape[-1]
    return n_samples, d_dim

def scale_data(
    x, 
    x_test, 
    y, 
    y_test, 
    transform_x: bool = True, 
    transform_y: bool = True, 
    scaler: str = 'std'
) -> tuple:
    if transform_x: 
        if scaler == 'std':
            mms = StandardScaler()
        elif scaler == 'minmax':
            mms = MinMaxScaler()
        else: 
            raise ValueError(f'Bad scaler: {scaler}')
        mms.fit(x)
        x, x_test = mms.transform(x), mms.transform(x_test)  
    if transform_y:
        y_mean, y_std = y.mean(), y.std()
        y, y_test = (y - y_mean) / y_std, (y_test - y_mean) / y_std
    return x, x_test, y, y_test

def load_transform_data(
    data_path, 
    test_size, 
    split_seed, 
    transform_x: bool = True, 
    transform_y: bool = True, 
    scaler: str = 'std'
) -> tuple:
    x, x_test, y, y_test = load_prepare_data(data_path, test_size, split_seed)
    return scale_data(x, x_test, y, y_test, transform_x, transform_y, scaler)

def get_batches(x, y, batch_size):
    num_samples = x.shape[0]
    indices = np.random.permutation(num_samples) 
    for start in range(0, num_samples, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield x[batch_idx], y[batch_idx]

def generate_x3_data(n_samples, d_dim, min_v=-4, max_v=4, std_err=3, seed=None, sort_x=True):
    rs = np.random.RandomState(seed)
    x = rs.uniform(min_v, max_v, (n_samples, d_dim))
    if sort_x:
        x = np.sort(x, axis=0)
    e = rs.normal(0.0, std_err, (n_samples,))
    y_true = (x**3).sum(axis=1)
    y = y_true + e
    return x, y, y_true
