"""
Verifies that _calculate_metrics returns correct c_index and c_index_ipcw,
matches direct sksurv calls, and handles edge cases properly.
"""
import sys
import types
import numpy as np
import pytest
from sksurv.metrics import concordance_index_censored, concordance_index_ipcw
from unittest.mock import MagicMock


def _import_wsi_metrics():
    """Import wsi_metrics while avoiding circular import chain.

    The circular import is:
    utils.wsi_metrics -> datasets.utils.continual_dataset -> datasets.__init__
    -> datasets.seq_survival -> utils.evaluate_wsi -> utils.evaluate_wsi_joint
    -> utils.wsi_metrics (partially initialized)

    We stub out datasets.utils.continual_dataset before importing.
    """
    stub = types.ModuleType("datasets.utils.continual_dataset")
    stub.ContinualDataset = type("ContinualDataset", (), {})
    sys.modules.setdefault("datasets.utils.continual_dataset", stub)

    import importlib
    wsi_metrics = importlib.import_module("utils.wsi_metrics")
    return wsi_metrics


wsi_metrics = _import_wsi_metrics()
_calculate_metrics = wsi_metrics._calculate_metrics
Surv = wsi_metrics.Surv
update_metric_dict_tuple = wsi_metrics.update_metric_dict_tuple
update_metric_matrix_tuple = wsi_metrics.update_metric_matrix_tuple
get_last_value_from_metric_dict_tuple = wsi_metrics.get_last_value_from_metric_dict_tuple


def _make_survival_train(event_times, censorships):
    """Helper to create survival_train structured array."""
    return Surv.from_arrays(
        event=(1 - censorships).astype(bool),
        time=event_times,
    )


def _make_mock_dataset(task_name="task_a", joint=False):
    """Create a mock dataset object for _calculate_metrics."""
    ds = MagicMock()
    ds.args = MagicMock()
    ds.args.joint_training = joint
    return ds


class TestCalculateMetrics:
    """Tests for _calculate_metrics function."""

    def test_returns_two_values(self):
        """Function should return exactly 2 values (c_index, c_index_ipcw)."""
        np.random.seed(42)
        n = 50
        risk_scores = np.random.rand(n)
        censorships = np.random.choice([0, 1], size=n, p=[0.7, 0.3]).astype(float)
        event_times = np.random.exponential(10, n)

        survival_train = _make_survival_train(event_times, censorships)
        ds = _make_mock_dataset()

        result = _calculate_metrics(ds, "task_a", survival_train, risk_scores, censorships, event_times)
        assert len(result) == 2, f"Expected 2 return values, got {len(result)}"

    def test_c_index_matches_sksurv(self):
        """c_index should match direct concordance_index_censored call."""
        np.random.seed(123)
        n = 100
        risk_scores = np.random.rand(n)
        censorships = np.random.choice([0, 1], size=n, p=[0.7, 0.3]).astype(float)
        event_times = np.random.exponential(10, n)

        survival_train = _make_survival_train(event_times, censorships)
        ds = _make_mock_dataset()

        c_index, c_index_ipcw = _calculate_metrics(ds, "task_a", survival_train, risk_scores, censorships, event_times)

        # Direct sksurv call
        expected_c_index = concordance_index_censored(
            (1 - censorships).astype(bool), event_times, risk_scores, tied_tol=1e-08
        )[0]

        assert c_index == pytest.approx(expected_c_index, abs=1e-10), \
            f"c_index mismatch: {c_index} vs {expected_c_index}"

    def test_c_index_ipcw_matches_sksurv(self):
        """c_index_ipcw should match direct concordance_index_ipcw call."""
        np.random.seed(456)
        n = 100
        risk_scores = np.random.rand(n)
        censorships = np.random.choice([0, 1], size=n, p=[0.7, 0.3]).astype(float)
        event_times = np.random.exponential(10, n)

        survival_train = _make_survival_train(event_times, censorships)
        survival_test = Surv.from_arrays(
            event=(1 - censorships).astype(bool), time=event_times
        )
        ds = _make_mock_dataset()

        _, c_index_ipcw = _calculate_metrics(ds, "task_a", survival_train, risk_scores, censorships, event_times)

        expected_ipcw = concordance_index_ipcw(survival_train, survival_test, estimate=risk_scores)[0]

        assert c_index_ipcw == pytest.approx(expected_ipcw, abs=1e-10), \
            f"c_index_ipcw mismatch: {c_index_ipcw} vs {expected_ipcw}"

    def test_nan_handling(self):
        """NaN risk scores should be filtered out correctly."""
        np.random.seed(789)
        n = 50
        risk_scores = np.random.rand(n)
        censorships = np.random.choice([0, 1], size=n, p=[0.7, 0.3]).astype(float)
        event_times = np.random.exponential(10, n)

        # Insert NaN at specific positions
        nan_indices = [5, 10, 20]
        risk_scores_with_nan = risk_scores.copy()
        risk_scores_with_nan[nan_indices] = np.nan

        survival_train = _make_survival_train(event_times, censorships)
        ds = _make_mock_dataset()

        c_index, c_index_ipcw = _calculate_metrics(
            ds, "task_a", survival_train, risk_scores_with_nan, censorships, event_times
        )

        # Compute expected with NaN removed
        clean_risk = np.delete(risk_scores, nan_indices)
        clean_censor = np.delete(censorships, nan_indices)
        clean_times = np.delete(event_times, nan_indices)

        expected_c_index = concordance_index_censored(
            (1 - clean_censor).astype(bool), clean_times, clean_risk, tied_tol=1e-08
        )[0]

        assert c_index == pytest.approx(expected_c_index, abs=1e-10), \
            f"c_index with NaN mismatch: {c_index} vs {expected_c_index}"

    def test_no_old_params_accepted(self):
        """_calculate_metrics should NOT accept all_risk_by_bin_scores parameter."""
        np.random.seed(42)
        n = 20
        risk_scores = np.random.rand(n)
        censorships = np.zeros(n)
        event_times = np.random.exponential(10, n)
        survival_train = _make_survival_train(event_times, censorships)
        ds = _make_mock_dataset()

        # This should raise TypeError since we removed the parameter
        with pytest.raises(TypeError):
            _calculate_metrics(
                ds, "task_a", survival_train, risk_scores, censorships, event_times,
                np.random.rand(n, 4)  # old all_risk_by_bin_scores
            )

    def test_perfect_discrimination(self):
        """When risk perfectly predicts events, c_index should be close to 1."""
        n = 50
        event_times = np.arange(1, n + 1, dtype=float)
        censorships = np.zeros(n)  # no censoring
        risk_scores = np.arange(n, 0, -1, dtype=float)  # higher risk = earlier event

        survival_train = _make_survival_train(event_times, censorships)
        ds = _make_mock_dataset()

        c_index, _ = _calculate_metrics(ds, "task_a", survival_train, risk_scores, censorships, event_times)
        assert c_index > 0.99, f"Perfect discrimination should give c_index near 1, got {c_index}"


class TestMetricTupleHelpers:
    """Tests for metric dict/matrix/tuple helper functions."""

    def test_update_metric_dict_tuple_2_elements(self):
        """update_metric_dict_tuple should work with 2-element tuples."""
        c_index_dict = {}
        c_index_ipcw_dict = {}
        metric_dict_tuple = (c_index_dict, c_index_ipcw_dict)
        metric_output_tuple = (0.75, 0.70)

        result = update_metric_dict_tuple("task_a", metric_output_tuple, metric_dict_tuple)
        assert len(result) == 2
        assert result[0]["val_task_a"] == 0.75
        assert result[1]["val_task_a"] == 0.70

    def test_get_last_value_2_elements(self):
        """get_last_value_from_metric_dict_tuple should work with 2-element tuples."""
        metric_dict_tuple = (
            {"val_task_a": 0.8, "val_task_b": 0.75},
            {"val_task_a": 0.7, "val_task_b": 0.65},
        )
        result = get_last_value_from_metric_dict_tuple(metric_dict_tuple)
        assert len(result) == 2
        assert result[0] == 0.75  # last value of first dict
        assert result[1] == 0.65  # last value of second dict

    def test_update_metric_matrix_tuple_2_elements(self):
        """update_metric_matrix_tuple should work with 2-element tuples."""
        c_index_matrix = {}
        c_index_ipcw_matrix = {}
        metric_matrix_tuple = (c_index_matrix, c_index_ipcw_matrix)
        metric_dict_tuple = (
            {"val_task_a": 0.8},
            {"val_task_a": 0.7},
        )

        result = update_metric_matrix_tuple("task_a", metric_dict_tuple, metric_matrix_tuple)
        assert len(result) == 2
        assert "train_task_a" in result[0]
        assert "train_task_a" in result[1]
