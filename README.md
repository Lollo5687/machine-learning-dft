
# Machine Learning in Density Functional Theory

**Convolutional Neural Networks for Kinetic Energy Functional Approximation**

This repository contains the Python implementation developed for my Bachelor's thesis in Physics, *"Apprendimento automatico nella teoria del funzionale della densità"* (Machine Learning in Density Functional Theory), completed at the University of Camerino.

## Overview

Density Functional Theory (DFT) provides a framework for studying quantum many-body systems using the particle density rather than the full many-body wavefunction. However, the exact form of the kinetic energy functional is generally unknown, representing a major challenge for Orbital-Free DFT.

The goal of this project is to investigate the use of **Convolutional Neural Networks (CNNs)** to approximate the kinetic energy functional of non-interacting fermionic systems.

The neural network is trained to learn the mapping:

```math id="6p6j78"
n(x) \longrightarrow T[n]
```

where $n(x)$ is the ground-state particle density and $T[n]$ is the corresponding kinetic energy.

Training data are generated from quantum systems confined to a one-dimensional region and subjected to external potentials constructed from randomly parameterized Gaussian functions.

The study considers systems of **1, 10, and 20 non-interacting fermions**, comparing different network architectures, preprocessing strategies, and input representations.

## Repository Structure

```text id="wzuhhu"
.
├── docs/
│   └── thesis (italian).pdf
├── results/
│   ├── images/             # Plots and visualizations
│   ├── logs/               # Training logs
│   └── weight/             # Saved model weights
├── src/
│   ├── data_processing/
│   │   └── Analisi_modelli.ipynb
│   ├── models/
│   │   └── cnns_1.py
│   └── training/
│       └── train.py
├── README.md
└── requirements.txt
```

### Source Code

The source code is organized into three main components:

- **`src/data_processing/Analisi_modelli.ipynb`** — Jupyter notebook used for data analysis and visualization of model performance.
- **`src/models/cnns_1.py`** — Contains the CNN implementations used to approximate the kinetic energy functional of the single-fermion system.
- **`src/training/train.py`** — Implements the model training procedure.

### Data and Results

- **`results/images/`** — Contains figures produced during the experiments, including ground-state density profiles, energy distributions, training and validation metrics (MSE and R²), and prediction error analysis.
- **`results/logs/`** — Stores training logs containing information about model configurations and training progress.
- **`results/weight/`** — Directory for storing trained model weights.
- **`docs/thesis (italian).pdf`** — Complete Bachelor's thesis, including theoretical background, methodology, and results for the 1, 10, and 20 fermion systems.

### Implementation Scope

The repository contains the code developed for the **single-fermion system**. The implementations for the 10 and 20 fermion systems are largely identical in structure, with differences in dataset preparation and preprocessing.

For this reason, the multi-fermion implementations are not included separately. However, their methodology and results are documented in the thesis.

## Methodology

### Neural Network Architecture

The model combines convolutional and pooling layers for feature extraction with fully connected layers for predicting kinetic energy.

Several architectures were investigated by varying the number of convolutional blocks and the dimensionality of the flattened representation.

For the single-fermion system, the best configuration, denoted **M5-60**, consists of five convolutional blocks with 60 features passed to the fully connected network.

The training procedure uses the Adam optimizer, the Softplus activation function, and the Mean Squared Error (MSE) loss.

### Multi-Fermion Systems

The thesis also investigates systems containing 10 and 20 non-interacting fermions.

For these systems, additional techniques were explored to improve generalization, including:

- Input and target standardization.
- Batch normalization and dropout.
- Learning-rate scheduling.
- Reduced fully connected network complexity.
- Alternative input/output representations, including the external potential as an additional input.

These experiments are discussed in the thesis but are not separately implemented in this repository.

## Results

The following results were obtained during the thesis and evaluated on independent test datasets.

| System | Model configuration | Test MSE [Ry²] | Test R² |
|---|---|---|---|
| 1 fermion | M5-60 | $3.287 * 10^{-7}$ | 0.9988 |
| 10 fermions | Modified M5-60, density only | $2.205 * 10^{-4}$ | 0.9784 |
| 20 fermions | Modified M5-60 | $3.226 * 10^{-4}$ | 0.989 |

### Main Finding

**Single-fermion system**

The CNN successfully approximated the kinetic energy functional, achieving predictions with errors generally below chemical accuracy, defined as $3.187 * 10^{-3}$. The results demonstrate that convolutional architectures can accurately learn the density-to-energy mapping for this system.

**Multi-fermion systems**

The extension to multiple fermions proved more challenging. Despite achieving relatively high R² values, the prediction errors remained significantly larger than in the single-fermion case.

Architectural improvements reduced overfitting and improved generalization, but did not fully resolve the accuracy limitations.

**Importance of the external potential**

For the 10-fermion system, providing both the particle density and the external potential as inputs significantly improved prediction accuracy.

The fraction of predictions within chemical accuracy increased from **18.9% to 53.3%**, suggesting that the external potential provides useful additional information for the learning process.

Overall, the results highlight both the potential of CNN-based kinetic energy functionals and the challenges associated with extending these approaches to more complex systems.

## Installation

Clone the repository and install the required Python dependencies:

```bash id="ptk09x"
git clone https://github.com/Lollo5687/machine-learning-dft.git
cd machine-learning-dft
pip install -r requirements.txt
```

The project requires Python and the libraries listed in `requirements.txt`, including PyTorch for neural network training.

The original datasets are not included in the repository and must be provided separately to reproduce the experiments.

### Training

To train a model, run the following command from the project root directory:

```bash
python -m src.training.train
```

The model architecture and training hyperparameters must be selected and configured directly in `src/training/train.py` before execution.

Different CNN architectures are defined in `src/models/cnns_1.py` and can be selected by modifying the training script.


## Thesis

For a detailed discussion of the theoretical background, dataset generation, neural network architectures, experimental results, and possible future developments, please refer to the complete Bachelor's thesis:

[**Apprendimento automatico nella teoria del funzionale della densità (PDF)**](docs/thesis%20%28italian%29.pdf)

**Lorenzo Ducci — University of Camerino, A.Y. 2024/2025**

Supervisor: **Prof. Sebastiano Pilati**

## Future Work

Possible extensions include investigating different classes of external potentials, improving the accuracy of multi-fermion models, and studying the use of learned functionals in variational optimization to recover ground-state densities and energies.
