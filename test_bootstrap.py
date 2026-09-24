"""Tests for bootstrap_sample function."""

import numpy as np
import pytest

from bootstrap import bootstrap_sample


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
