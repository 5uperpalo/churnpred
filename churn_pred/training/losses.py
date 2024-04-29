import itertools
from typing import List, Tuple, Literal, Callable

import numpy as np
import lightgbm as lgb
from scipy.misc import derivative
from lightv.training.utils import _sigmoid, predict_cls_lgbm_from_raw
from lightv.training.metrics import (
    lgbm_f1,
    lgbm_recall,
    lgbm_accuracy,
    lgbm_precision,
)


def set_feval_fobj(
    n_class: int, loss: Literal["focal_loss", None]
) -> Tuple[Callable, Callable]:
    """Helper method to set custom evaluation and objective function.

    Args:
        n_class (int): number of classes in the dataset
        loss (str): type of loss function to use
            * 'None' - default for given task
            * 'focal_loss' - focal loss
    Returns:
        feval (Callable): custom evaluation function
        fobj (Callable): custom objective function
    """
    if loss == "focal_loss":
        feval = lgbm_focal_feval(n_class)
        fobj = lgbm_focal_fobj(n_class)
    else:
        feval, fobj = None, None
    return feval, fobj


def lgbm_focal_feval(
    n_class: int,
) -> Callable:
    """Function to adjust Focal loss to be used as evaluation in LightGBM.
    feval is accompanied with other f1, precision, recall, accuracy for each class.

    Args:
        n_class (int): number of classes in the dataset
    Returns:
        feval(List[FunctionType]): focal loss evaluation functions for LightGBM
    """

    def lgbm_focal_feval_n_class(
        preds_raw: np.ndarray, lgbDataset: lgb.Dataset
    ) -> List[Tuple[str, float, bool]]:
        y_true = lgbDataset.get_label()
        task: Literal["binary", "multiclass"] = (
            "binary" if n_class == 2 else "multiclass"
        )
        if n_class > 2:
            preds_raw_reshaped = preds_raw.reshape(-1, n_class, order="F")
        else:
            preds_raw_reshaped = preds_raw
        preds = predict_cls_lgbm_from_raw(preds_raw=preds_raw_reshaped, task=task)
        return list(
            itertools.chain(
                *[
                    [lgbm_focal_loss_eval(preds_raw, y_true, 0.25, 1.0, n_class)],
                    [lgbm_f1(y_true, preds)],
                    [lgbm_precision(y_true, preds)],
                    [lgbm_recall(y_true, preds)],
                    [lgbm_accuracy(y_true, preds)],
                    [lgbm_f1(y_true, preds, label=i) for i in range(n_class)],
                    [lgbm_precision(y_true, preds, label=i) for i in range(n_class)],
                    [lgbm_recall(y_true, preds, label=i) for i in range(n_class)],
                ]
            )
        )

    return lgbm_focal_feval_n_class


def lgbm_focal_fobj(
    n_class: int,
) -> Callable:
    """Function to adjust Focal loss to be used as objective in LightGBM.

    Args:
        n_class (int): number of classes in the dataset
    Returns:
        fobj(FunctionType): focal loss objective functions for LightGBM
    """

    def lgbm_focal_fobj_n_class(
        preds_raw: np.ndarray, lgbDataset: lgb.Dataset
    ) -> Tuple[float, float]:
        y_true = lgbDataset.get_label()
        return lgbm_focal_loss(
            preds_raw=preds_raw, y_true=y_true, alpha=0.25, gamma=1.0, n_class=n_class
        )

    return lgbm_focal_fobj_n_class


def lgbm_focal_loss(
    preds_raw: np.ndarray, y_true: np.ndarray, alpha: float, gamma: float, n_class: int
) -> Tuple[float, float]:
    """Adapation of the Focal Loss for lightgbm to be used as training loss.
    See original paper:
    * https://arxiv.org/pdf/1708.02002.pdf
    and custom training loss documentation:
    * https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.train.html

    Args:
        y_pred (ndarray): array with the predictions
        y_true (ndarray): array with the ground truth
        alpha (float): loss function variable
        gamma (float): loss function variable
    Returns:
        grad (float): The value of the first order derivative (gradient) of the loss with
            respect to the elements of preds for each sample point.
        hess (float): The value of the second order derivative (Hessian) of the loss with
            respect to the elements of preds for each sample point.
    """
    # N observations x num_class arrays
    if n_class > 2:
        y_true = np.eye(n_class)[y_true.astype("int")]
        y_pred = preds_raw.reshape(-1, n_class, order="F")
    else:
        y_pred = preds_raw.astype("int")

    def partial_fl(x):
        return _focal_loss(x, y_true, alpha, gamma)

    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    if n_class > 2:
        return grad.flatten("F"), hess.flatten("F")
    else:
        return grad, hess


def lgbm_focal_loss_eval(
    preds_raw: np.ndarray, y_true: np.ndarray, alpha: float, gamma: float, n_class: int
) -> Tuple[str, float, bool]:
    """Adapation of the Focal Loss for lightgbm to be used as evaluation loss.
    See original paper https://arxiv.org/pdf/1708.02002.pdf

    Args:
        y_pred (ndarray): array with the predictions
        y_true (ndarray): array with the ground truth
        alpha (float): loss function variable
        gamma (float): loss function variable
    Returns:
        result (tuple): tuple containing name of the loss function,
            its value and bool value for LighGBM (is_higher_better)
    """
    # N observations x num_class arrays
    if n_class > 2:
        y_true = np.eye(n_class)[y_true.astype("int")]
        y_pred = preds_raw.reshape(-1, n_class, order="F")
    else:
        y_pred = preds_raw

    loss = _focal_loss(y_pred, y_true, alpha, gamma)
    result = ("focal_loss", np.mean(loss), False)
    return result


def _focal_loss(
    y_pred: np.ndarray, y_true: np.ndarray, alpha: float, gamma: float
) -> np.ndarray:
    """Helpter function for lgbm_focal_loss and lgbm_focal_loss_eval

    Args:
        y_pred (ndarray): array with the predictions
        y_true (ndarray): array with the ground truth
        alpha (float): loss function variable
        gamma (float): loss function variable
    Returns:
        loss (ndarray): focal loss
    """
    preds = _sigmoid(y_pred)
    loss = (
        -(alpha * y_true + (1 - alpha) * (1 - y_true))
        * ((1 - (y_true * preds + (1 - y_true) * (1 - preds))) ** gamma)
        * (y_true * np.log(preds) + (1 - y_true) * np.log(1 - preds))
    )
    return loss
