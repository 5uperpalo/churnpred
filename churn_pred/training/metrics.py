from typing import Tuple, Union, Optional

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import (
    f1_score,
    ndcg_score,
    recall_score,
    accuracy_score,
    precision_score,
    mean_squared_error,
)


def lgbm_f1(
    actual: np.ndarray,
    predicted: np.ndarray,
    label: Optional[int] = None,
) -> Tuple[str, float, bool]:
    """Implementation of the f1 score to be used as evaluation score for lightgbm
    see feval [documentation](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html).  # noqa
    The adaptation is required since when using custom losses
    the row prediction needs to passed through a sigmoid to represent a
    probability.

    Args:
        actual (np.ndarray): actual values
        predicted (np.ndarray): predicted values
        label: Union[None, int]=None: specify individual class for metric,
            if None - weighted average of all classes
    Returns:
        result (tuple): tuple containing name of the score, its value and bool value for
            LighGBM (is_higher_better)
    """
    # 0 is treated in Python as None
    if type(label) == int:
        result = (
            "f1_" + str(label),
            f1_score(actual, predicted, average=None, zero_division=0)[label],
            True,
        )
    else:
        result = (
            "f1",
            f1_score(actual, predicted, average="weighted", zero_division=0),
            True,
        )
    return result


def lgbm_precision(
    actual: np.ndarray,
    predicted: np.ndarray,
    label: Union[None, int] = None,
) -> Tuple[str, float, bool]:
    """Implementation of the precision score to be used as evaluation
    score for lightgbm. The adaptation is required since when using custom losses
    the row prediction needs to passed through a sigmoid to represent a
    probability.

    Args:
        actual (np.ndarray): actual values
        predicted (np.ndarray): predicted values
        label: Union[None, int]=None: specify individual class for metric,
            if None - weighted average of all classes
    Returns:
        result (tuple): tuple containing name of the score, its value and bool value for
            LighGBM (is_higher_better)
    """
    # 0 is treated in Python as None
    if type(label) == int:
        result = (
            "precision_" + str(label),
            precision_score(actual, predicted, average=None, zero_division=0)[label],
            True,
        )
    else:
        result = (
            "precision",
            precision_score(actual, predicted, average="weighted", zero_division=0),
            True,
        )
    return result


def lgbm_recall(
    actual: np.ndarray,
    predicted: np.ndarray,
    label: Union[None, int] = None,
) -> Tuple[str, float, bool]:
    """Implementation of the recall score to be used as evaluation
    score for lightgbm. The adaptation is required since when using custom losses
    the row prediction needs to passed through a sigmoid to represent a
    probability.

    Args:
        actual (np.ndarray): actual values
        predicted (np.ndarray): predicted values
        label: Union[None, int]=None: specify individual class for metric,
            if None - weighted average of all classes
    Returns:
        result (tuple): tuple containing name of the score, its value and bool value for
            LighGBM (is_higher_better)
    """
    # 0 is treated in Python as None
    if type(label) == int:
        result = (
            "recall_" + str(label),
            recall_score(actual, predicted, average=None, zero_division=0)[label],
            True,
        )
    else:
        result = (
            "recall",
            recall_score(actual, predicted, average="weighted", zero_division=0),
            True,
        )
    return result


def lgbm_accuracy(actual: np.ndarray, predicted: np.ndarray) -> Tuple[str, float, bool]:
    """Implementation of the accuracy score to be used as evaluation
    score for lightgbm. The adaptation is required since when using custom losses
    the row prediction needs to passed through a sigmoid to represent a
    probability.

    Args:
        actual (np.ndarray): actual values
        predicted (np.ndarray): predicted values
    Returns:
        result (tuple): tuple containing name of the score, its value and bool value for
            LighGBM (is_higher_better)
    """
    result = ("accuracy", accuracy_score(actual, predicted), True)
    return result


def smape(actual: np.ndarray, predicted: Union[np.ndarray, list]) -> float:
    """Symmetric Mean Absolute Percentage Error
    https://vedexcel.com/how-to-calculate-smape-in-python/

    Args:
        actual (np.ndarray): actual values
        predicted (np.ndarray): predicted values
    Returns:
        smape (float): symmetric mean absolute percentage error
    """
    return (
        100
        / len(actual)
        * np.sum(2 * np.abs(predicted - actual) / (np.abs(actual) + np.abs(predicted)))
    )


def ndcg(actual: np.ndarray, predicted: Union[np.ndarray, list]) -> float:
    """Normalized Discounted Cumulative Gain

    Args:
        actual (np.ndarray): actual values
        predicted (np.ndarray): predicted values
    Return:
        ndcg (float): normalized discounted cumulative gain
    """
    return ndcg_score(
        [rankdata(actual, method="ordinal")],
        [rankdata(predicted, method="ordinal")],
    )


def rmse(actual: np.ndarray, predicted: Union[np.ndarray, list]) -> float:
    """Root Mean Squared Error

    Args:
        actual (np.ndarray): actual values
        predicted (np.ndarray): predicted values
    Returns:
        rmse (float): root mean square error
    """
    return mean_squared_error(actual, predicted, squared=False)


def nacil(
    actual: np.ndarray,
    high_quantile_predicted: np.ndarray,
    low_quantile_predicted: np.ndarray,
):
    """Normalized Average Confidence Interval Length for Quantile Regression.
    The value is normalized to actual(expected) value.

    Args:
        actual (np.ndarray): actual values
        high_quantile_predicted (np.ndarray): high quantile predicted values
        low_quantile_predicted (np.ndarray): low quantile predicted values
    Returns:
        nacil (float): normalized average confidence interval length
    """
    return np.mean((high_quantile_predicted - low_quantile_predicted) / actual)


def aiqc(actual: np.ndarray, high_quantile: np.ndarray, low_quantile: np.ndarray):
    """Average InterQuantile Coverage for Quantile Regression.
    Check if the interquantile coverage is close to the coverage of in the test dataset.

    Args:
        actual (np.ndarray): actual values
        high_quantile_predicted (np.ndarray): high quantile predicted values
        low_quantile_predicted (np.ndarray): low quantile predicted values
    Returns:
        aiqc (float): average interquantile coverage
    """
    return np.sum((actual < high_quantile) & (actual > low_quantile)) / len(actual)
