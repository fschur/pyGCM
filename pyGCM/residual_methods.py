import numpy as np
from sklearn.linear_model import LinearRegression
import GPy


def linear_regression_residuals(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Computes residuals of Y on X using ordinary linear regression.

    Args:
        Y (np.ndarray): Target array, shape (n_samples,) or (n_samples, dY).
        X (np.ndarray): Covariate array, shape (n_samples, dX).

    Returns:
        np.ndarray: Residuals, same shape as Y.
    """
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    residuals = np.zeros_like(Y)
    for col_idx in range(Y.shape[1]):
        model = LinearRegression()
        model.fit(X, Y[:, col_idx])
        y_pred = model.predict(X)
        residuals[:, col_idx] = Y[:, col_idx] - y_pred

    if residuals.shape[1] == 1:
        return residuals.ravel()
    return residuals


def gp_regression_residuals(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Computes residuals of Y on X using a Gaussian Process regression
    with GPy's RBF kernel.

    Args:
        Y (np.ndarray): Target array, shape (n_samples,) or (n_samples, dY).
        X (np.ndarray): Covariate array, shape (n_samples, dX).

    Returns:
        np.ndarray: Residuals, same shape as Y.
    """
    # Ensure Y is 2D
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_samples, dY = Y.shape
    residuals = np.zeros_like(Y)

    # Fit a separate GP for each output dimension
    for col_idx in range(dY):
        y_col = Y[:, col_idx:col_idx+1]  # shape (n,1)

        # Define a GPy RBF kernel
        kernel = GPy.kern.RBF(
            input_dim=X.shape[1],
            variance=1.0,
            lengthscale=1.0
        )

        # Create GPRegression model
        model = GPy.models.GPRegression(X, y_col, kernel=kernel, normalizer=True)
        model.optimize(optimizer="lbfgs")

        # Predict on training data
        mean_pred, _ = model.predict(X)

        # Residual = Y - prediction
        residuals[:, col_idx] = (y_col - mean_pred).ravel()

    # If only one output dimension, return a 1D array
    if dY == 1:
        return residuals.ravel()

    return residuals


def group_mean_residuals(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Computes residuals of Y on X by taking the group mean within each
    unique 'category' or 'category-combination' of X and subtracting it.

    E.g., for each row i, residual_i = Y_i - mean(Y_j for all j s.t. X_j == X_i).

    This is useful when X is purely categorical (or a set of categorical columns).

    Args:
        Y (np.ndarray): Target array, shape (n_samples,) or (n_samples, dY).
        X (np.ndarray): Categorical array, shape (n_samples, dX) or (n_samples,).

    Returns:
        np.ndarray: Residuals, shape same as Y.
    """
    # Ensure Y is 2D for consistent logic
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    n_samples, dY = Y.shape

    # If X is 1D, reshape to (n,1) to unify the logic
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Convert each row of X into a tuple => group index
    # np.unique(..., axis=0, return_inverse=True) will give us a group index (group_idx)
    # for each row i, indicating which unique row of X it matches.
    _, group_idx = np.unique(X, axis=0, return_inverse=True)

    # We'll store residuals in the same shape as Y
    residuals = np.zeros_like(Y)

    # For each distinct group, compute mean(Y) in that group and subtract
    for grp in np.unique(group_idx):
        mask = (group_idx == grp)
        # Mean for each Y column
        grp_mean = Y[mask].mean(axis=0)
        residuals[mask] = Y[mask] - grp_mean

    # If Y was originally 1D, reduce back to 1D
    if dY == 1:
        return residuals.ravel()
    return residuals
