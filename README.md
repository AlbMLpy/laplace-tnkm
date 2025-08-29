Linearized Laplace Approximation in Bayesian Tensor Network Kernel Machines
=====

## ✨ Project Description
Uncertainty estimation is critical for robust decision making in the presence of ambiguous or out-of-distribution inputs.
Gaussian Process (GP) is a classic kernel-based approach that provides uncertainty quantification and performs well on small to medium scale datasets.
Alternatively, formulating the primal learning problem under tensor network assumptions leads to scalable tensor network kernel machines.
This reformulation raises a fundamental question: *how can tensor network kernel machines be extended for probabilistic inference and uncertainty quantification?*

We propose *a novel Bayesian Tensor Network Kernel Machine (LA-TNKM)* that employs *a (linearized) Laplace approximation for Bayesian inference*. A comprehensive set of numerical experiments shows that the proposed method consistently matches or surpasses Gaussian Processes and Bayesian Neural Networks (BNNs) across diverse UCI regression benchmarks, highlighting both its effectiveness and practical relevance. 

## 📊 Datasets
For real-data experiments, we use the following nine UCI regression datasets (Dua and Graff, 2017):
|  | Boston | Concrete | Energy | Kin8nm | Naval | Power | Protein | Red Wine | Yacht |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------| -------------| -------------|
| **N - Sample Size** | 506 | 1030 | 768 | 8192 | 11934 | 9568 | 45730 | 1588 | 308 |
| **D - Data Dim.** | 13 | 8 | 8 | 8 | 16 | 4 | 9 | 11 | 6 |

## ⚙️ Environment
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

## 🚀 How to Reproduce the Numerical Experiments

0. Create and activate the virtual environment (Environment section).

1. **Run:**
   ```shell
   python setup_project.py
   ```
   to download 9 datasets and configure the project directories. 

2. Once the setup script has completed, **run:**
    ```shell
    cd experiments
    ```
    This folder contains three subdirectories, each corresponding to a distinct experiment described in the paper: `ablation_study`, `uncertainty_synthetic` and `uci_regression`.

3. `ablation_study`
    -  **Run:** `cd ablation_study`
    -  **Train**: 
        ```bash
        python training.py
        ```
        Computes evaluation metrics and predictions for further comparison. These are stored in `artifacts/training_artifacts`.
    -  **Analyze:** Run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate figures stored in `artifacts/results`.

4. `uncertainty_synthetic`
    -  **Run:** `cd uncertainty_synthetic`
    -  **Analyze:** Run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate figures stored in `artifacts`.

5. `uci_regression`:
    -  **Run:** `cd uci_regression`
    -  **Train:** 
        ```bash
        python training.py model 'all'
        ```
        Computes evaluation metrics and predictions for further comparison. These are stored in `artifacts/training_artifacts`. Use `python training.py -h` to see all options (e.g., parallel/sequential mode and `n_jobs`).
    -  **Analyze:** Run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate the final table for comparison.


## 📜 Citation

If you find our work helpful, please consider citing the paper:

```bibtex
In Progress!
```
