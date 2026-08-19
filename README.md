# incremental-rs

Out-of-Core and Incremental Model Fitting for Rust.

## Overview
`incremental-rs` fills the gap in the Rust ML ecosystem where estimators expect fully in-memory dataset arrays. It provides `partial_fit` implementations matching scikit-learn's incremental capability scope.

## Estimator Capabilities
| Estimator | Incremental Algorithm | Trait |
| :--- | :--- | :--- |
| `IncrementalLinearRegression` | Stochastic Gradient Descent (MSE) | `IncrementalSupervisedEstimator` |
| `IncrementalLogisticRegression` | SGD / Cross-Entropy / OvR | `IncrementalSupervisedEstimator` |
| `IncrementalGaussianNaiveBayes` | Welford's Online Means/Variance | `IncrementalSupervisedEstimator` |
| `MiniBatchKMeans` | Sculley (2010) Web-Scale K-Means | `IncrementalUnsupervisedEstimator` |

## Out-of-Scope Design
Decision trees, Random Forests, and exact closed-form Ordinary Least Squares (OLS) do not have direct incremental variants and are intentionally excluded from online fitting scope.


##Licensed under MIT.