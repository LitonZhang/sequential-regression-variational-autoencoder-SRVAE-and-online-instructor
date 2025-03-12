# Core Algorithms for "A modeling and adaptive evolution method for simulation parameters of digital twin shop floor"

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository contains the official implementation of algorithms proposed in the paper **"A modeling and adaptive evolution method for simulation parameters of digital twin shop floor"** published in *Robotics and Computer-Integrated Manufacturing*. If this repository aids your research, please cite our work (see [Citation](#citation)).

## Features
- **Sequential Regression Variational Autoencoder (SRVAE)**: A novel time series forecasting model for digital twin parameter evolution.
- **Online Instructor Algorithm**: Adaptive online training framework for dynamic shop floor environments.
- **Benchmark Models**:
  - LSTM
  - Deep Factors:doi={https://doi.org/10.48550/arXiv.1905.12417}
  - Static probability
  - Dynamic probability:doi={https://doi.org/10.1080/00207543.2022.2051088}
- Modular implementation with PyTorch

## Training
Configure training mode using command-line arguments in main.py
Arguments:
-p: Enable pre-training mode
-o: Enable online adaptation mode

## Testing
Evaluate models using model_test.py

## Citation
If you use this codebase in your research, please cite our original paper:
@article{yourcitationkey,
  title={A modeling and adaptive evolution method for simulation parameters of digital twin shop floor},
  author={Author1, Author2, ...},
  journal={Robotics and Computer-Integrated Manufacturing},
  year={2023},
  volume={xx},
  pages={xxx--xxx},
  doi={10.1016/j.rcim.xxxx.xxxxxx}
}
