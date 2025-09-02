import os
import zipfile
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path('data')
DATA2ID = {'energy': 242, 'concrete': 165, 'power': 294}
UCI_PART = 'https://archive.ics.uci.edu/static/public/'

def load_data():
    load_yacht()
    load_boston()
    load_kin8nm()
    load_protein()
    load_naval()
    load_wine_red()
    for data_name in DATA2ID.keys():
        load_uci(data_name)

def load_uci(data_name: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    from ucimlrepo import fetch_ucirepo  
    dataset = fetch_ucirepo(id=DATA2ID[data_name]) 
    x, y = dataset.data.features, dataset.data.targets 
    yv = y.values[:, 0] if y.ndim > 1 else y.values
    x = x.assign(target=yv)
    x.to_csv(DATA_DIR / f'{data_name}.csv', index=None)
    print(f'{data_name.capitalize()} Dataset downloaded ✅')

def _load_url(url: str, name_data: str) -> str:
    # Create data dir if not exists:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temp_file = DATA_DIR / 'temp.zip'
    urllib.request.urlretrieve(url, filename=temp_file)
    print(f'{name_data.capitalize()} Dataset downloaded ✅')
    return temp_file

def load_airline(path_raw: str):
    def get_airline_df(
        path, 
    ):
        return (
            pd.read_csv(path, header=0, index_col=0)
            .drop('Year', axis=1)
            .assign(ArrTime=lambda x: 60*np.floor(x.ArrTime/100) + np.mod(x.ArrTime, 100))
            .assign(DepTime=lambda x: 60*np.floor(x.DepTime/100) + np.mod(x.DepTime, 100))
            .rename({'ArrDelay': 'target'}, axis=1)
            .reset_index(drop=True)
            .astype(int)
        )
    get_airline_df(path_raw).to_csv(DATA_DIR / 'airline.csv', header=True, index=False)

def load_banana():
    url = (
        'https://raw.githubusercontent.com/' 
         + 'SaravananJaichandar/MachineLearning/master/' 
         + 'Standard%20Classification%20Dataset/banana/banana.csv'
    )
    data_path = DATA_DIR / 'banana.csv'
    temp_file = _load_url(url, 'Banana')
    os.rename(temp_file, data_path)
    pd.read_csv(data_path).rename({'Class': 'class'}, axis=1).to_csv(data_path, index=None)

def load_yacht():
    url = UCI_PART + '243/yacht+hydrodynamics.zip'
    temp_file = _load_url(url, 'Yacht')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    temp_file2 = DATA_DIR / 'yacht_hydrodynamics.data'
    with open(temp_file2, 'r') as f:
        lines = [[float(v) for v in line.rstrip('\n').split(' ') if v != ''] for line in f]
    pd.DataFrame(
        lines[:-1], 
        columns=['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'target']
    ).to_csv(DATA_DIR / 'yacht.csv', index=None)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    temp_file2.unlink(missing_ok=True)

def load_boston():
    url = (
        "https://archive.ics.uci.edu/ml/"
        + "machine-learning-databases/housing/housing.data"
    )
    temp_file = _load_url(url, 'Boston')
    with open(temp_file, 'r') as f:
        lines = [[float(v) for v in line.rstrip('\n').split(' ') if v != ''] for line in f]
    pd.DataFrame(
        lines, 
        columns=[*[f'f{i}' for i in range(1, 14)], 'target']
    ).to_csv(DATA_DIR / 'boston.csv', index=None)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)

def load_kin8nm():
    url = "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.csv"
    temp_file = _load_url(url, 'Kin8nm')
    data_path = DATA_DIR / 'kin8nm.csv'
    os.rename(temp_file, data_path)
    pd.read_csv(data_path).rename({'y': 'target'}, axis=1).to_csv(data_path, index=None)

def load_protein():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv"
    temp_file = _load_url(url, 'protein')
    data_path = DATA_DIR / 'protein.csv'
    os.rename(temp_file, data_path)
    pd.read_csv(data_path).rename({'RMSD': 'target'}, axis=1).to_csv(data_path, index=None)

def load_naval():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip"
    temp_file = _load_url(url, 'naval')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        with zip_ref.open("UCI CBM Dataset/data.txt") as f:
            lines = [[float(v) for v in line.decode("utf-8").rstrip('\n').split(' ') if v != ''] for line in f]
    df = pd.DataFrame(
        lines, 
        columns=[*[f'f{i}' for i in range(1, 17)], 'target', 'target_2']
    )
    df.drop('target_2', axis=1).to_csv(DATA_DIR / 'naval.csv', index=None)
    temp_file.unlink(missing_ok=True)

def load_wine_red():
    url = UCI_PART + '186/wine+quality.zip'
    temp_file = _load_url(url, 'wine')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    df = pd.read_csv(DATA_DIR / 'winequality-red.csv', sep=';')
    df = df.rename({'quality': 'target'}, axis=1)
    for file_name in os.listdir(DATA_DIR):
        if 'wine' in file_name:
            file_path = os.path.join(DATA_DIR, file_name)
            try: os.remove(file_path) 
            except: pass
    temp_file.unlink(missing_ok=True)
    df.to_csv(DATA_DIR / 'wine_red.csv', index=None)

def load_airfoil(): 
    url = UCI_PART + '291/airfoil+self+noise.zip'
    temp_file = _load_url(url, 'Airfoil')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR)
    temp_file2 = DATA_DIR / 'airfoil_self_noise.dat'
    with open(temp_file2) as f:
        lines = [[float(v) for v in line.rstrip('\n').split('\t') if v != ''] for line in f]
    pd.DataFrame(
        lines[:-1], 
        columns=['f1', 'f2', 'f3', 'f4', 'f5', 'target']
    ).to_csv(DATA_DIR / 'airfoil.csv', index=None)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    temp_file2.unlink(missing_ok=True)

def load_spambase():
    url = UCI_PART + '94/spambase.zip'
    temp_file = _load_url(url, 'Spambase')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR, members=['spambase.data'])
    temp_file2 = DATA_DIR / 'spambase.data'
    df = pd.read_csv(temp_file2, header=None, names=[f"f{i}" for i in range(1, 58)] + ['class'])
    df.loc[df['class'] == 0, 'class'] = -1
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    temp_file2.unlink(missing_ok=True)
    df.to_csv(DATA_DIR / 'spambase.csv', index=None)

def load_census_income():
    url = UCI_PART + '2/adult.zip'
    temp_file = _load_url(url, 'Censis Income')
    with zipfile.ZipFile(temp_file, 'r') as zip_ref:
        zip_ref.extractall(DATA_DIR, members=['adult.data', 'adult.test'])
    tf2, tf3 = DATA_DIR / 'adult.data', DATA_DIR / 'adult.test'
    columns = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 
        'marital-status', 'occupation', 'relationship', 'race', 'sex',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target'
    ]
    df = pd.read_csv(tf2, names=columns, na_values=['?',' ?'])
    df = pd.concat(
        [df, pd.read_csv(tf3, names=columns, na_values=['?',' ?'])], 
        ignore_index=True, 
        sort=False, 
        axis=0
    )
    df['target'] = df['target'].str.rstrip('.')
    df = df.dropna()
    continuous_labels = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_labels = list(set(set(columns)-set(continuous_labels)))
    categorical_labels.append(categorical_labels.pop(categorical_labels.index('target')))
    df = pd.get_dummies(df, prefix=categorical_labels, columns=categorical_labels, drop_first=True)
    df = df.rename({'target_ >50K': 'class'}, axis=1)
    mask = df.columns[df.dtypes == 'bool']
    df[mask] = df[mask].astype(int)
    df.loc[df['class'] == 0, 'class'] = -1
    df.to_csv(DATA_DIR / 'census_income.csv', header=True, index=False)
    # Remove temp file:
    temp_file.unlink(missing_ok=True)
    tf2.unlink(missing_ok=True)
    tf3.unlink(missing_ok=True) 

def in_docker():
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt") as f:
            for line in f:
                if "docker" in line or "kubepods" in line:
                    return True
    except FileNotFoundError:
        pass
    return False

if __name__ == '__main__':
    load_data()
    if in_docker():
        print("🐋 You are running in Docker. Preparing analysis.py files.")
        exp_dir = "./experiments/"
        for nb_dir in ['ablation_study', 'uci_regression', 'uncertainty_synthetic']:
            subprocess.run(
                ["jupyter", "nbconvert", "--to", "script", 'analysis.ipynb'],
                cwd=exp_dir + nb_dir,
                check=True
            )
    print("🎉 Setup complete.")
