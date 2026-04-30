<div align="center">

<h1>Laplace Approximation for Bayesian Tensor Network Kernel Machines</h1>

<b> Authors: </b> <a href="https://albmlpy.github.io/">Albert Saiapin</a>, Kim Batselier

[ArXiv](https://arxiv.org/abs/2604.26673) | [Experiments](https://github.com/AlbMLpy/laplace-tnkm/tree/main#-how-to-reproduce-the-numerical-experiments)

<p align="center">
  <a href="https://arxiv.org/abs/2604.26673"><img src="assets/compare_hess.png?raw=True" alt="Figure 1" width="90%"></a>
</p>

**Figure 1:** Test NLL performance of LA-TNKM with Linearized Laplace Approximation (LLA) under different Hessian approximations, as a function of the thresholding hyperparameter $\hat{t}$ (log-scale, normalized by the largest Hessian eigenvalue). Results across real-world datasets show that performance is sensitive to the Hessian approximation, while the Last and Block variants are most consistently strong on average.

</div>


## 💫 Project Description
This repository accompanies our paper on **LA-TNKM**, *a Bayesian Tensor Network Kernel Machine* based on *a (linearized) Laplace approximation* for scalable uncertainty estimation.

The method enables principled posterior inference while preserving the efficiency of tensor network models. Experiments on UCI regression benchmarks show that LA-TNKM consistently matches or outperforms Gaussian Processes and Bayesian Neural Networks (BNNs). See the paper for full theoretical details and derivations.


## 📈 Datasets
For real-data experiments, we use the following nine UCI regression datasets (Dua and Graff, 2017):

|  | Boston | Concrete | Energy | Kin8nm | Naval | Power | Protein | Red Wine | Yacht |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------| -------------| -------------|
| **N** | 506 | 1030 | 768 | 8192 | 11934 | 9568 | 45730 | 1588 | 308 |
| **D** | 13 | 8 | 8 | 8 | 16 | 4 | 9 | 11 | 6 |

where **N** is the training sample size and **D** is the data dimensionality.

## ⚙️ Environment
We use `conda` package manager to install required python packages. Once `conda is installed`, **run** the following command (while in the root of the repository):
```shell
conda env create -f environment.yml
```
This will create a new environment named `bayes_env` with all required packages already installed. You can **install** additional packages by running:
```shell
conda install <package name>
```

In order to read and run `Jupyter Notebooks` you may follow either of two options:
1. [*recommended*] using notebook-compatibility features of IDEs, e.g. via `python` and `jupyter` extensions of [VS Code](https://code.visualstudio.com/).
2. install jupyter notebook packages:
  either with `conda install jupyterlab` or with `conda install jupyter notebook`

## 🐳 (Optional) Running Experiments with Docker

Instead of setting up a Conda environment manually, you can run the entire experiment inside a Docker container. This ensures full reproducibility and requires only `Docker to be installed` on your system. 

ℹ️ Note: Depending on your Docker installation, you may need to prefix
all `docker` commands in this guide with `sudo`. 

1. From the project root (where the `Dockerfile` is located) **build** the Docker image:
    ```shell
    docker build -t la-tnkm-project .
    ```

2. **Run** the container interactively:
    ```shell
    docker run -it -v $(pwd)/experiments:/app/experiments --name la-tnkm la-tnkm-project
    ```
3. **You are all set to reproduce the Numerical Experiments!** 🤗

4. **Re-enter** the same container:
    ```shell
    docker start -ai la-tnkm
    ```

5. **Cleaning up** (optional):
    
    1. Remove the container:
        ```shell
        docker rm la-tnkm
        ```
    2. Remove the image:
        ```shell
        docker rmi la-tnkm-project
        ```

## 🚀 How to Reproduce the Numerical Experiments

0. **Activate** the virtual environment:
    ```shell
    conda activate bayes_env
    ```

1. **Run:**
   ```shell
   python setup_project.py
   ```
   to download all datasets and configure the project directories. 

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
    -  **Analyze:** If using `Docker`, run: `python analysis.py`. Otherwise, run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate figures stored in `artifacts/results`.

4. `uncertainty_synthetic`
    -  **Run:** `cd uncertainty_synthetic`
    -  **Analyze:** If using `Docker`, run: `python analysis.py`. Otherwise, run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate figures stored in `artifacts`.

5. `uci_regression`:
    -  **Run:** `cd uci_regression`
    -  **Train:** 
        ```bash
        python training.py model 'all'
        ```
        Computes evaluation metrics and predictions for further comparison. These are stored in `artifacts/training_artifacts`. Use `python training.py --help` to see all options (e.g., parallel/sequential mode and `n_jobs`).
    -  **Analyze:** If using `Docker`, run: `python analysis.py`. Otherwise, run `analysis.ipynb` in VS Code using the `bayes_env` environment, or open it with `jupyter lab` to generate the final table for comparison.


## 📜 Citation

If you find this repository useful in your work, please cite the accompanying paper:

```bibtex
@misc{saiapin2026laplaceapproximationbayesiantensor,
      title={Laplace Approximation for Bayesian Tensor Network Kernel Machines}, 
      author={Albert Saiapin and Kim Batselier},
      year={2026},
      eprint={2604.26673},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2604.26673}, 
}
```
