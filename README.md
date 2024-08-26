
# RSNA 2024 Lumbar Spine Degenerative Classification

This repository contains the code for participating in the RSNA 2024 Lumbar Spine Degenerative Classification competition on Kaggle. The goal of the competition is to develop a model that can accurately classify lumbar spine degeneration from MRI images.

## Table of Contents
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)

## Project Structure

The repository is structured as follows:

```
.
├── config.py          # Configuration file for paths and hyperparameters
├── dataset.py         # Dataset and data loading utilities
├── inference.py       # Script for running inference on test data
├── net.py             # Neural network model definitions
├── train.py           # Main training script
├── train_model.py     # Utility script for training models with different configurations
├── utils.py           # Utility functions for logging, metrics, etc.
└── README.md          # Project documentation
```

## Installation

To set up the environment for running this code, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/rsna-2024-lumbar-spine.git
   cd rsna-2024-lumbar-spine
   ```

2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train the model, use the `train.py` script. It supports various configurations, which can be adjusted in the `config.py` file.

```bash
python train.py 
```

### Inference

To run inference on new data, use the `inference.py` script:

```bash
python inference.py
```
