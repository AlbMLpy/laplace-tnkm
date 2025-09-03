SCALER = 'std'
SPLIT_SEED = 1
TEST_SIZE = 0.1
TRANSFORM_X = True
TRANSFORM_Y = True
MODELS = ['cpr_la', 'cpr_lla']
DATASETS = ['yacht', 'energy', 'concrete', 'wine_red']

RANK = 10
SEED = 13
M_ORDER = 8
BETA_E = 1.0
GAMMA_W = 1e-6
ALS_EPOCHS = 100
PD_SAMPLE_SEED = 1
HESS_THS = [1e-5, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
HESS_TYPES = ['full', 'gauss_newton', 'block', 'mf', 'last']
