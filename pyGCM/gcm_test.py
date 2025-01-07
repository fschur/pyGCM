import numpy as np
import scipy.stats as st
from sklearn.preprocessing import OneHotEncoder

# Import your three residual methods
from .residual_methods import (
    linear_regression_residuals,
    gp_regression_residuals,
    group_mean_residuals,
)


class GCMTest:
    """
    A class to perform the Generalized Correlation Measure (GCM) test.
    Supports optional one-hot encoding of E, X, Y if they are categorical.
    If 'get_resid_method' is 'categorical', we compute group-mean residuals.

    Usage:
        gcm = GCMTest(get_resid='linear')  # or 'GP', 'categorical', or custom callable
        test_stat, p_val = gcm.fit(E, X, Y, categorical_X=True, ...)
    """

    def __init__(self, get_resid='linear'):
        """
        Args:
            get_resid (str or callable):
                - 'linear': Use linear regression residuals
                - 'GP': Use Gaussian process regression residuals
                - 'categorical': Use group-mean residuals (for X considered categorical)
                - callable: custom function, signature get_resid(Y, X) -> residual_array
        """
        self.get_resid_method = get_resid
        self.test_stat_ = None
        self.p_value_ = None

    def fit(self,
            E: np.ndarray,
            X: np.ndarray,
            Y: np.ndarray,
            resid_Y: np.ndarray = None,
            resid_E: np.ndarray = None,
            categorical_E: bool = False,
            categorical_X: bool = False,
            categorical_Y: bool = False) -> tuple[float, float]:
        """
        Tests the Null-Hypothesis that
            E indep Y | X
        using the GCM test.

        If resid_Y or resid_E are None/empty, they are computed based on
        the get_resid_method set in the constructor.

        If 'get_resid' is 'categorical' (or a custom function that does group means),
        then we rely on that to handle X being purely categorical.

        Args:
            E (np.ndarray): Environment variable(s), shape (n_samples,) or (n_samples, dE).
            X (np.ndarray): Covariates, shape (n_samples, dX).
            Y (np.ndarray): Target variable(s), shape (n_samples,) or (n_samples, dY).
            resid_Y (np.ndarray): Optional precomputed residuals for Y given X.
            resid_E (np.ndarray): Optional precomputed residuals for E given X.
            categorical_E (bool): If True, treat E as discrete (possible one-hot).
            categorical_X (bool): If True, treat X as discrete (possible one-hot).
            categorical_Y (bool): If True, treat Y as discrete (possible one-hot).

        Returns:
            (float, float): (test_statistic, p_value)
        """
        # If X is truly categorical and user set get_resid to 'linear' or 'GP', it might be ill-defined.
        # Optionally, you can check for that scenario:
        if categorical_X and self.get_resid_method in ['linear', 'GP']:
            raise ValueError(
                "X is specified as categorical, but 'linear' or 'GP' "
                "residual methods do not make sense for purely categorical X. "
                "Use 'categorical' or a custom callable."
            )

        # Possibly one-hot or verify one-hot for E
        if categorical_E:
            E = self._maybe_one_hot_encode(E)

        if categorical_Y:
            Y = self._maybe_one_hot_encode(Y)

        E, Y = self._to_2_dim(E), self._to_2_dim(Y)

        # If residuals not provided, compute them
        if resid_Y is None or resid_Y.size == 0:
            resid_Y = self._dispatch_residuals(Y, X)

        if resid_E is None or resid_E.size == 0:
            resid_E = self._dispatch_residuals(E, X)

        resid_Y, resid_E = self._to_2_dim(resid_Y), self._to_2_dim(resid_E)

        n_samples = Y.shape[0]

        # Outer product: shape => (n, dY, dE) => flatten => (n, dY*dE)
        resid_product = (resid_Y[:, :, np.newaxis] *
                         resid_E[:, np.newaxis, :]).reshape(n_samples, -1)

        test_stat, p_val = self._compute_gcm_statistic_and_p_value(resid_product, n_samples)
        self.test_stat_ = test_stat
        self.p_value_ = p_val
        return test_stat, p_val

    @staticmethod
    def _to_2_dim(arr: np.ndarray) -> np.ndarray:
        """Ensures arr is 2D: (n,1) if originally (n,)."""
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def _dispatch_residuals(self, Y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """
        Computes residuals based on 'get_resid_method'
        (i.e., 'linear', 'GP', 'categorical', or a custom callable).

        Args:
            Y (np.ndarray): shape (n, dY)
            X (np.ndarray): shape (n, dX)

        Returns:
            np.ndarray: Residuals, shape (n, dY) or (n,) if single-dimensional.
        """
        if callable(self.get_resid_method):
            return self.get_resid_method(Y, X)

        elif isinstance(self.get_resid_method, str):
            method_lower = self.get_resid_method.lower()
            if method_lower == 'linear':
                return linear_regression_residuals(Y, X)
            elif method_lower == 'gp':
                return gp_regression_residuals(Y, X)
            elif method_lower == 'categorical':
                return group_mean_residuals(Y, X)
            else:
                raise ValueError(
                    f"Unknown residual method '{self.get_resid_method}'. "
                    "Use 'linear', 'GP', 'categorical', or provide a callable."
                )
        else:
            raise TypeError(
                "'get_resid' must be either a string ('linear', 'GP', 'categorical') "
                "or a callable function."
            )

    @staticmethod
    def _compute_gcm_statistic_and_p_value(resid_product: np.ndarray,
                                           n_samples: int) -> tuple[float, float]:
        """
        Computes the GCM test statistic and p-value from residual products.

        GCM statistic:
          stat = max( |(sqrt(n)*mean(rp)) / sqrt(E[rp^2] - (E[rp])^2)| )
        p-value is two-sided from Normal(0,1):
          p_val = 2 * (1 - Φ(|stat|))

        Args:
            resid_product (np.ndarray): shape (n_samples, k)
            n_samples (int): number of samples

        Returns:
            (float, float): (test_stat, p_value)
        """
        rp_mean = resid_product.mean(axis=0)
        nom = np.sqrt(n_samples) * rp_mean
        denom = np.sqrt((resid_product ** 2).mean(axis=0) - rp_mean ** 2)

        test_stat = np.max(np.abs(nom / denom))
        p_value = 2 * (1 - st.norm.cdf(abs(test_stat)))
        return test_stat, p_value

    def _maybe_one_hot_encode(self, arr: np.ndarray) -> np.ndarray:
        """
        If arr is 1D => one-hot encode it.
        If arr is 2D => check if it is already one-hot. If not, encode it.

        Returns:
            np.ndarray: Possibly one-hot-encoded array (2D).
        """
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
            return self._one_hot_encode(arr)

        if self._is_already_one_hot(arr):
            return arr
        else:
            return self._one_hot_encode(arr)

    @staticmethod
    def _is_already_one_hot(arr: np.ndarray) -> bool:
        """
        Check if arr is strictly one-hot: all in {0,1} and each row sums to 1.
        """
        if not np.all(np.isin(arr, [0, 1])):
            return False
        row_sums = arr.sum(axis=1)
        return np.allclose(row_sums, 1)

    @staticmethod
    def _one_hot_encode(arr: np.ndarray) -> np.ndarray:
        """
        One-hot encodes a 2D array (n_samples, 1) or (n_samples, d).
        """
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        return encoder.fit_transform(arr)

