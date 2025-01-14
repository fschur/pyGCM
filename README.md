[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-1804.07203-b31b1b.svg)](https://arxiv.org/abs/1804.07203)

# pyGCM: A Python Implementation of the Generalized Covariance Measure

This repository provides code for the conditional independence testing method proposed in the paper:   
[*The Hardness of Conditional Independence Testing and the Generalised Covariance Measure*](https://arxiv.org/abs/1804.07203).  

In particular, the **pyGCM** package implements the Generalized Covariance Measure (GCM) for testing conditional independence. It includes multiple residual-computation methods:

- **Linear Regression Residuals**
- **Gaussian Process Regression Residuals**
- **Group-Mean Residuals** (for categorical conditioning variables)

## Table of Contents
1. [Introduction](#introduction)  
2. [Installation](#installation)  
3. [Usage](#usage)  
4. [Testing](#testing)
5. [References](#references)  

---

## Introduction

It is a classical result in statistical theory that testing conditional independence $(E \perp Y \mid X)$ is a hard problem, especially when $X$ is high-dimensional or continuous. This code implements the **Generalized Covariance Measure (GCM)**, as proposed in the cited paper, to address this difficulty.  
 
The key idea of **GCM** is:
1. Regress $E$ on $X$ to obtain residuals $\hat{\varepsilon}_E$.
2. Regress $Y$ on $X$ to obtain residuals $\hat{\varepsilon}_Y$.
3. Compute a test statistic based on the sample covariance between $\hat{\varepsilon}_E$ and $\hat{\varepsilon}_Y$.  

---

## Installation
Run

```bash
pip install -e . 
```

If you want to run additional experiments, or test out all features (e.g., Gaussian Process regression, group-mean residuals for categorical variables), install the extra dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

Below is a minimal example of how to use the **pyGCM** package. The primary entry point is the `GCMTest` class.

```python
import numpy as np
from pyGCM import GCMTest

# Generate some synthetic data
n = 500
X = np.random.randn(n, 1)
E = 0.5 * X + 0.1 * np.random.randn(n, 1)  # 'Environment' or confound
Y = 0.8 * X + 0.1 * np.random.randn(n, 1)

# Initialize GCM with linear residuals
gcm = GCMTest(get_resid='linear')

# Compute the test statistic and the p-value
test_stat, p_val = gcm.fit(E, X, Y)

print("GCM statistic:", test_stat)
print("p-value:", p_val)
```

**Key arguments** to `GCMTest(...).fit(...)` include:
- `E, X, Y`: The variables of interest (numpy arrays).
- `get_resid`: A string or callable specifying how to compute residuals:
  - `'linear'`: Linear regression
  - `'GP'`: Gaussian process regression
  - `'categorical'`: Group-mean residuals for categorical `X`
- `categorical_X`, `categorical_E`, `categorical_Y`: Booleans indicating if the variables should be one-hot-encoded.

---

## Testing

The repository contains a set of **PyTest**-based unit tests under the `tests/` directory (e.g., `tests/test_gcm.py`). To run them:

```bash
pytest tests
```

These tests:
- Generate synthetic data for each residual method (`linear`, `GP`, `categorical`).
- Check that the computed p-value is within a reasonable tolerance of a reference value.

---

## References
The methodology implemented in this package is based on the paper:

> Shah, R. D., & Peters, J. (2018).  
> The Hardness of Conditional Independence Testing and the Generalised Covariance Measure.  
> [arXiv:1804.07203](https://arxiv.org/abs/1804.07203)

If you use **pyGCM** in your own work, please cite the original paper by Shah and Peters.

---

**License**  
This project is licensed under the terms of the [MIT License](https://opensource.org/licenses/MIT).  

For questions or issues, please open an [issue](https://github.com/YourUserName/pyGCM/issues) or send a pull request. We welcome contributions!
