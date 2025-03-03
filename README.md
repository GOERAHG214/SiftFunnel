# SiftFunnel

This repository contains the official implementation of the paper:  
**"How breakable is privacy: probing and resisting model inversion attack in collaborative inference"**  
[Arxiv Link](https://arxiv.org/abs/2501.00824)

## Overview
This project provides implementations for different types of Model Inversion Attacks (MIA) and defense mechanisms. The key components in this repository include:

- **Gen-based MIA** (Generative-based Model Inversion Attack):
  - `train_inversion.py`: Training script for generative-based MIA.
  - `inversion_model_packages.py`: Package containing inversion models used for this attack.

- **MINE**:
  - `Mutual_Information_estimator.py`: Script to train the MINE.
  - `T_model_packages.py`: Package containing models used for MINE estimator.

- **Target Neural Network Training**:
  - `train_classifier.py`: Script for training target neural networks.
  - `target_model_packages.py`: Package containing target neural network architectures.

- **MLE-based MIA (White-box Attack)**:
  - `white_box_attack.py`: Implementation of Maximum Likelihood Estimation (MLE)-based MIA.

- **Dataset Utility Functions**:
  - `utilis.py`: Utility functions for dataset processing and manipulation.

- **SiftFunnel Defense Mechanism**:
  - `information_siftfunnel.py`: Script for training models protected using the SiftFunnel method.
  - `info_packages.py`: Package containing SiftFunnel-related model components.

## Installation
To set up the environment, install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
### Training a Target Model
```bash
python train_classifier.py
```

### Performing Model Inversion Attack (Gen-based MIA)
```bash
python train_inversion.py
```

### Performing Mutual Information Neural Estimator
```bash
python Mutual_Information_estimator.py
```

### Running the SiftFunnel Defense Mechanism
```bash
python information_siftfunnel.py
```

## Citation
If you use this code, please cite our paper:
```bibtex
@article{siftfunnel2025,
  author    = {anonymous},
  title     = {How breakable is privacy: probing and resisting model inversion attack in collaborative inference},
  journal   = {ArXiv},
  year      = {2025},
  url       = {https://arxiv.org/abs/2501.00824}
}
```
