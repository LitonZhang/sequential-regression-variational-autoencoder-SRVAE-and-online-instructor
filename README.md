# Core Algorithms for "A modeling and adaptive evolution method for simulation parameters of digital twin shop floor"

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of algorithms proposed in the paper **"A modeling and adaptive evolution method for simulation parameters of digital twin shop floor"** submited in **Robotics and Computer-Integrated Manufacturing**. If this repository aids your research, please cite our work (see [Citation](#citation)).

## Features
- **Sequential Regression Variational Autoencoder (SRVAE)**: A novel time series forecasting model.
- **Online Instructor Algorithm**: An online learning algorithm for machine learning.
- **Benchmark Models**:
  - LSTM
  - Deep Factors  doi=https://doi.org/10.48550/arXiv.1905.12417
  - Static probability
  - Dynamic probability  doi=https://doi.org/10.1080/00207543.2022.2051088
- Modular implementation with PyTorch

## Training
Configure training mode using arguments in main.py
- **Arguments**:
  - -p: Enable pre-training mode
  - -o: Enable online-training mode

## Testing
Evaluate models using model_test.py

## Citation
If you use this codebase in your research, please cite our original paper:doi=https://doi.org/10.1016/j.rcim.2025.103090
