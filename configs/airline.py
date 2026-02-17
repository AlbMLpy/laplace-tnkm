from typing import Optional
from dataclasses import dataclass

RANKS = [5, 20]
N_SAMPLES = [10000, 100000, 1000000, 5929413]

@dataclass
class Config:
    n_trials: int = 10
    res_name: str = 'results'
    transform_x: bool = True
    transform_y: bool = True
    scaler: str = 'minmax'
    rank: Optional[int] = None # To be defined during run time
    gamma_w: Optional[float] = None # To be defined during run time
    alpha: Optional[float] = None # To be defined during run time
    m_order: int = 40
    init_type: str = 'k_mtx'
    beta_e: Optional[float] = None
    n_epoch_tkrr: int = 10
    n_epoch_la_tnkm: int = 1
    n_epoch_vi: int = 1
    pd_samples: int = 5
    beta_e_samples: int = 5
    seed: int = 0
