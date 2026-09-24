import numpy as np

def bootstrap_sample(data, compute_stat, n_bootstrap=1000):
    """
    Generate the bootstrap distribution of a statistic

    Parameters
    ----------
    data : array-like
        original sample (for regression: 2D array with columns [x, y])

    compute_stat : callable
        function that computes a univariate statistic from data

    n_bootstrap : int, default 1000
        number of boostrap replicates to generate

    Returns
    -------
    numpy.ndarray
        Array of bootstrap statistics, length n_bootstrap

    Raises
    ------
    ValueError
        If data is empty, n_bootstrap < 1, or data has wrong shape
    TypeError
        If compute_stat is not callable

    Example
    -------
    TBA
    """
    pass


def bootstrap_ci(bootstrap_stats, alpha=0.05):
    """
    Calculate a CI from bootstrap distribution.

    Parameters
    ----------
    bootstrap_stats : numpy.ndarray
        Bootstrap statistics from bootstrap_sample(...).

    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of the CI.

    Raises
    ------
    ValueError
        If alpha not in (0, 1) or if bootstrap_stats is empty.

    Example
    -------
    >>> stats = np.array([1, 2, 3, 4, 5])
    >>> lower, upper = bootstrap_ci(stats)
    >>> lower <= upper
    True
    """
    bootstrap_stats = np.asarray(bootstrap_stats)

    # Check that bootstrap distribution is not empty.
    if bootstrap_stats.size == 0:
        raise ValueError("bootstrap_stats cannot be empty")

    # Check that alpha is a valid significance level.
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    lower = np.quantile(
        bootstrap_stats,
        alpha / 2,
    )

    upper = np.quantile(
        bootstrap_stats,
        1 - alpha / 2,
    )

    return float(lower), float(upper)



def r_squared(data):
    """
    Calculate R^2 from a linear regression.

    Parameters
    ----------
    data : array-like, shape (n, 2)
        Data with columns [x, y].

    Returns
    -------
    float
        R-squared value between 0 and 1.

    Raises
    ------
    ValueError
        If data doesn't have exactly 2 columns or < 2 rows.

    Example
    -------
    >>> data = np.array([
    ...     [1, 3],
    ...     [2, 5],
    ...     [3, 7],
    ... ])
    >>> r_squared(data)
    1.0
    """
    data = np.asarray(data, dtype=float)

    # Data must be a two-dimensional array with exactly two columns.
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            "data must be a 2D array with exactly two columns"
        )

    # At least two observations are required.
    if data.shape[0] < 2:
        raise ValueError(
            "data must contain at least two rows"
        )

    x = data[:, 0]
    y = data[:, 1]

    # Fit y = beta_0 + beta_1 * x using least squares.
    design_matrix = np.column_stack(
        (np.ones(len(x)), x)
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix,
        y,
        rcond=None,
    )

    fitted_values = design_matrix @ coefficients

    # Residual sum of squares.
    ss_res = np.sum(
        (y - fitted_values) ** 2
    )

    # Total sum of squares.
    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    if ss_tot == 0:
        raise ValueError(
            "R-squared is undefined when y has zero variance"
        )

    result = 1 - ss_res / ss_tot

    # Protect against tiny numerical errors such as
    # 1.0000000000000002 or -1e-16.
    result = np.clip(
        result,
        0.0,
        1.0,
    )

    return float(result)
