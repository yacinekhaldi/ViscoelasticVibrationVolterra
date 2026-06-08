# Learning Memory Kernels in Second-Order Volterra Models

## Overview

Modeling viscoelastic vibration remains a persistent challenge due to memory effects, coupled dynamics, and the difficulty of linking data-driven models with physical interpretability. Conventional approaches, such as Prony-series identification and fractional calculus formulations, can capture relaxation behavior but often rely on handcrafted parameterizations and may be sensitive to numerical discretization.

This repository contains the official implementation of the paper:

> **Learning Memory Kernels in Second-Order Volterra Models: A Data-Driven Approach for Viscoelastic Systems**  
> *Mechanics of Time-Dependent Materials*

We introduce a **physics-informed neural framework** that combines classical hereditary mechanics with modern machine learning to learn and forecast viscoelastic dynamics directly from data while preserving physical consistency.

---

## Key Features

- **Neural Prony State-Space Layer (NPSSL)** for differentiable memory modeling.
- **Physics-informed learning** through governing equation residuals.
- **Temporal Convolutional Network (TCN)** encoder for causal feature extraction.
- **Multi-task learning framework** for forecasting, classification, and parameter estimation.
- Support for both **synthetic** and **real viscoelastic datasets**.

---

## Methodology

### Neural Prony State-Space Layer (NPSSL)

The NPSSL provides a differentiable realization of the linear Volterra memory operator and generalizes the classical Prony representation. Memory effects are represented through internal causal states:

\[
z_m[n]
\]

which evolve recursively over time. To ensure physical passivity and stability, Prony amplitudes are constrained to remain nonnegative using a **softplus** parameterization.

### Physics-Informed Neural Architecture

The proposed framework combines:

1. A **Temporal Convolutional Network (TCN)** encoder for extracting temporal features.
2. An **NPSSL memory layer** for reconstructing the hereditary memory term \(H(t)\).
3. Multiple prediction heads for downstream tasks.

### Multi-Task Learning

The network jointly learns:

- **Multi-step displacement forecasting**
- **Kernel-family classification**
- **Prony-amplitude regression**

### Physics Residual Loss

A physics-based residual is incorporated into the loss function to enforce consistency with the governing second-order viscoelastic oscillator equation. This residual balances:

- Inertial forces
- Stiffness forces
- External forcing
- Viscoelastic memory stress

---

## Installation

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducing the Results

### 1. Generate Synthetic Datasets

Create forcing signals and kernel manifests:

```bash
python viscoelastic_vibration_datasets_generator.py
```

### 2. Compute Numerical Solutions

Generate ground-truth trajectories using the numerical solver:

```bash
python numerical_solver.py
```

### 3. Train the Model

Train the NPSSL framework on synthetic or real LPCID datasets:

```bash
python ml_solver.py
```

---

## Dataset

The repository supports:

- Synthetic viscoelastic vibration datasets generated from second-order Volterra systems.
- Real-world LPCID data for experimental validation.

Additional dataset instructions are available in the `data/` directory.

---

## Results

The proposed framework demonstrates:

- Accurate multi-step forecasting of viscoelastic responses.
- Reliable identification of memory-kernel families.
- Estimation of physically meaningful Prony coefficients.
- Improved interpretability through explicit memory-state modeling.

Generated metrics, predictions, and plots are saved in the `results/` directory.

---

## Citation

If you use this repository in your research, please cite:

```bibtex
@article{khaldi2026learning,
  title   = {Learning Memory Kernels in Second-Order Volterra Models: A Data-Driven Approach for Viscoelastic Systems},
  author  = {Khaldi, Yacine; Belaifa, Meriem; Benzaoui, Amir},
  journal = {Mechanics of Time-Dependent Materials},
  year    = {2026},
  doi     = {DOI_HERE}
}
```

---

## License

This project is distributed under the **MIT License**.

See the [LICENSE](LICENSE) file for more information.

---

## Contact

For questions, collaborations, or issues related to the implementation, please open a GitHub issue or contact the authors.
