The Linearized Laplace Approximation in Bayesian Tensor Networks
=====

## Project Description:
In progress ...

## Datasets:
In progress ...

## Environment
We use `conda` package manager to install required python packages. Once `conda is installed`, run the following command (while in the root of the repository):
```
conda env create -f environment/environment.yaml
```
This will create a new environment named `bayes_env` with all required packages already installed. You can install additional packages by running:
```
conda install <package name>
```
To activate the virtual environment:
```
conda activate bayes_env
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `conda install jupyterlab` or with `conda install jupyter notebook`

## Reproducing the Numerical Experiments:

0. Create and activate the virtual environment (Environment section).

1. Run:
   ```shell
   python load_data.py
   ```
   to load all the datasets locally and configure internal directories. 

2. In progress ...
