import numpy as np
from lightv.training.metrics import (
    ndcg,
    rmse,
    smape,
    lgbm_f1,
    lgbm_recall,
    lgbm_accuracy,
    lgbm_precision,
)

np.random.seed(0)
preds_binary = np.random.randint(low=0, high=2, size=5, dtype=int)
y_true = np.random.randint(low=0, high=2, size=5, dtype=int)


def test_lgbm_f1():
    metric = lgbm_f1(
        actual=y_true,
        predicted=preds_binary,
        label=None,
    )
    assert all(
        [
            (metric[0] == "f1"),
            np.isclose(metric[1], 0.75),
            metric[2] is True,
        ]
    )


def test_lgbm_precision():
    metric = lgbm_precision(
        actual=y_true,
        predicted=preds_binary,
        label=None,
    )
    assert all(
        [
            (metric[0] == "precision"),
            np.isclose(metric[1], 1.0),
            metric[2] is True,
        ]
    )


def test_lgbm_recall():
    metric = lgbm_recall(
        actual=y_true,
        predicted=preds_binary,
        label=None,
    )
    assert all(
        [
            (metric[0] == "recall"),
            np.isclose(metric[1], 0.6),
            metric[2] is True,
        ]
    )


def test_lgbm_accuracy():
    metric = lgbm_accuracy(actual=y_true, predicted=preds_binary)
    assert all(
        [
            (metric[0] == "accuracy"),
            np.isclose(metric[1], 0.6),
            metric[2] is True,
        ]
    )


def test_smape():
    metric = smape(actual=y_true, predicted=preds_binary)
    assert np.isclose(metric, 80.0)


def test_ndcg():
    metric = ndcg(actual=y_true, predicted=preds_binary)
    assert np.isclose(metric, 0.973755976697597)


def test_rmse():
    metric = rmse(actual=y_true, predicted=preds_binary)
    assert np.isclose(metric, 0.6324555320336759)
