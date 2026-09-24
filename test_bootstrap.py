import numpy as np
import pytest

from bootstrap import bootstrap_sample, bootstrap_ci, r_squared


def test_integration_all_three_functions():
    x = np.arange(
        1,
        11,
        dtype=float,
    )

    y = 2 * x + 1

    data = np.column_stack(
        (x, y)
    )

    np.random.seed(123)

    stats = bootstrap_sample(
        data,
        r_squared,
        n_bootstrap=200,
    )

    lower, upper = bootstrap_ci(stats)

    assert stats.shape == (200,)

    # Every resampled point still lies on y = 2x + 1.
    assert np.allclose(
        stats,
        1.0,
    )

    assert lower == pytest.approx(1.0)
    assert upper == pytest.approx(1.0)

# ============================================================
# Student B: Ziyi Kong
# ============================================================

# ============================================================
# Tests for bootstrap_sample
# ============================================================

def test_bootstrap_sample_returns_correct_shape():
    """Return one statistic for each bootstrap replicate."""
    data = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ])

    def compute_mean_y(sample):
        return np.mean(sample[:, 1])

    result = bootstrap_sample(
        data,
        compute_mean_y,
        n_bootstrap=100,
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (100,)


def test_bootstrap_sample_preserves_sample_size():
    """Each bootstrap replicate should preserve the original sample size."""
    data = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
        [4.0, 8.0],
    ])

    def sample_size(sample):
        return len(sample)

    result = bootstrap_sample(
        data,
        sample_size,
        n_bootstrap=20,
    )

    assert np.all(result == len(data))


def test_bootstrap_sample_preserves_rows():
    """Resampling should preserve paired x-y observations."""
    data = np.array([
        [1.0, 101.0],
        [2.0, 102.0],
        [3.0, 103.0],
        [4.0, 104.0],
    ])

    def pairs_are_valid(sample):
        return float(
            np.all(sample[:, 1] - sample[:, 0] == 100)
        )

    result = bootstrap_sample(
        data,
        pairs_are_valid,
        n_bootstrap=50,
    )

    assert np.all(result == 1.0)


def test_bootstrap_sample_empty_data_raises_value_error():
    """Empty input should raise ValueError."""
    data = np.empty((0, 2))

    with pytest.raises(ValueError):
        bootstrap_sample(
            data,
            np.mean,
            n_bootstrap=10,
        )


@pytest.mark.parametrize("n_bootstrap", [0, -1, -10])
def test_bootstrap_sample_invalid_n_bootstrap(n_bootstrap):
    """n_bootstrap must be at least one."""
    data = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    with pytest.raises(ValueError):
        bootstrap_sample(
            data,
            np.mean,
            n_bootstrap=n_bootstrap,
        )


def test_bootstrap_sample_noncallable_stat_raises_type_error():
    """compute_stat must be callable."""
    data = np.array([
        [1.0, 2.0],
        [2.0, 4.0],
    ])

    with pytest.raises(TypeError):
        bootstrap_sample(
            data,
            5,
            n_bootstrap=10,
        )


def test_bootstrap_sample_wrong_shape_raises_value_error():
    """Regression data must have exactly two columns."""
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    with pytest.raises(ValueError):
        bootstrap_sample(
            data,
            np.mean,
            n_bootstrap=10,
        )


# ============================================================
# Student A: Yiwu Gu
# ============================================================


# ============================================================
# Tests for bootstrap_ci
# ============================================================

def test_bootstrap_ci_returns_tuple():
    """bootstrap_ci should return a tuple containing two bounds."""
    bootstrap_stats = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0
    ])

    result = bootstrap_ci(bootstrap_stats)

    assert isinstance(result, tuple)
    assert len(result) == 2


def test_bootstrap_ci_bounds_are_ordered():
    """The lower confidence bound should not exceed the upper bound."""
    bootstrap_stats = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0
    ])

    lower, upper = bootstrap_ci(bootstrap_stats)

    assert lower <= upper


def test_bootstrap_ci_bounds_within_statistic_range():
    """Confidence interval bounds should lie within the bootstrap distribution."""
    bootstrap_stats = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0
    ])

    lower, upper = bootstrap_ci(bootstrap_stats)

    assert lower >= np.min(bootstrap_stats)
    assert upper <= np.max(bootstrap_stats)


@pytest.mark.parametrize("alpha", [0, 1, -0.1, 1.1])
def test_bootstrap_ci_invalid_alpha_raises_value_error(alpha):
    """alpha must lie strictly between zero and one."""
    bootstrap_stats = np.array([
        1.0, 2.0, 3.0, 4.0, 5.0
    ])

    with pytest.raises(ValueError):
        bootstrap_ci(
            bootstrap_stats,
            alpha=alpha,
        )


def test_bootstrap_ci_empty_stats_raises_value_error():
    """Empty bootstrap statistics should raise ValueError."""
    bootstrap_stats = np.array([])

    with pytest.raises(ValueError):
        bootstrap_ci(bootstrap_stats)


# ============================================================
# Tests for r_squared
# ============================================================

def test_r_squared_perfect_positive_relationship():
    """A perfect positive linear relationship should have R-squared equal to one."""
    data = np.array([
        [1.0, 3.0],
        [2.0, 5.0],
        [3.0, 7.0],
        [4.0, 9.0],
    ])

    result = r_squared(data)

    assert result == pytest.approx(1.0)


def test_r_squared_perfect_negative_relationship():
    """A perfect negative linear relationship should also have R-squared equal to one."""
    data = np.array([
        [1.0, 8.0],
        [2.0, 6.0],
        [3.0, 4.0],
        [4.0, 2.0],
    ])

    result = r_squared(data)

    assert result == pytest.approx(1.0)


def test_r_squared_returns_value_between_zero_and_one():
    """R-squared should be a float between zero and one."""
    data = np.array([
        [1.0, 1.0],
        [2.0, 2.5],
        [3.0, 2.0],
        [4.0, 5.0],
    ])

    result = r_squared(data)

    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0


def test_r_squared_zero_linear_relationship():
    """Data with zero linear association should have R-squared equal to zero."""
    data = np.array([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 2.0],
        [4.0, 1.0],
    ])

    result = r_squared(data)

    assert result == pytest.approx(0.0)


def test_r_squared_wrong_number_of_columns_raises_value_error():
    """Input data must have exactly two columns."""
    data = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    with pytest.raises(ValueError):
        r_squared(data)


def test_r_squared_one_column_raises_value_error():
    """Input data with fewer than two columns should raise ValueError."""
    data = np.array([
        [1.0],
        [2.0],
        [3.0],
    ])

    with pytest.raises(ValueError):
        r_squared(data)


def test_r_squared_fewer_than_two_rows_raises_value_error():
    """Input data must contain at least two rows."""
    data = np.array([
        [1.0, 2.0],
    ])

    with pytest.raises(ValueError):
        r_squared(data)
