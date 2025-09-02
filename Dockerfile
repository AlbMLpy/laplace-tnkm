# Base image with Miniconda
FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy conda environment file and create environment
COPY environment.yml .
RUN conda env create -f environment.yml

# Make conda environment active in all commands
SHELL ["conda", "run", "-n", "bayes_env", "/bin/bash", "-c"]

# Copy your project code and data
COPY . .
