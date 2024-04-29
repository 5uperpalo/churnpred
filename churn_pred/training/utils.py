from typing import List, Tuple, Literal, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset as lgbDataset


def to_lgbdataset(
    train: pd.DataFrame,
    cat_cols: List[str],
    target_col: str,
    id_cols: List[str] = [],
    valid: Optional[pd.DataFrame] = None,
) -> Tuple[lgbDataset, Optional[lgbDataset]]:
    """Transform pandas dataframe to lgbm dataset, or datasets(eval).

    Args:
        train (pd.DataFrame):
            training dataset
        cat_cols (list):
            list of categorical columns
        target_col (str):
            target column in the dataset
        id_cols (list):
            list of identifier columns
        valid (pd.DataFrame):
            validation dataset
    Returns:
        lgb_train (lgbDataset):
            lgbm training dataset
        lgb_valid (lgbDataset):
            lgbm valid dataset
    """

    lgb_train = lgbDataset(
        train.drop(columns=[target_col] + id_cols),
        train[target_col],
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    if valid is not None:
        lgb_valid = lgbDataset(
            valid.drop(columns=[target_col] + id_cols),
            valid[target_col],
            reference=lgb_train,
            free_raw_data=False,
        )
    else:
        lgb_valid = None

    return lgb_train, lgb_valid


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Apply sigmoid function to an array.

    Args:
        x (ndarray): 1D numpy array
    """
    sigmoided_array = 1.0 / (1.0 + np.exp(-x))
    return sigmoided_array


def _binary_margin_to_prob(x: np.ndarray) -> np.ndarray:
    """Transform single element np array with binary classificaiton probability
    to 2 element array for analysis purposes.

    Args:
        x (ndarray):
            1D single element numpy array
    """
    transformed = np.array([1 - x[0], x[0]])
    return transformed


def _softmax(x: np.ndarray) -> np.ndarray:
    """Apply softmax to an array.

    Args:
        x (ndarray):
            1D numpy array
    """
    exp_x = np.exp(x - np.max(x))
    softmaxed = exp_x / (np.sum(exp_x, axis=0, keepdims=True) + 1e-6)
    return softmaxed


def predict_proba_lgbm_from_raw(
    preds_raw: np.ndarray,
    task: Literal["binary", "multiclass"],
    binary2d: Optional[bool] = False,
) -> np.ndarray:
    """Apply softmax to array of arrays that is an output of lightgbm.predct().
    This replaces predict_proba().

    Args:
        predicted_raw (ndarray):
            1D numpy array of arrays
        task (str):
            type of task/objective
        binary2d (boolean):
            wether the output of binary classification should be 1 or 2d vector
    Returns:
        predicted_probs (ndarray): array with predicted probabilities
    """
    if task == "binary":
        predicted_probs = _sigmoid(preds_raw)
        if binary2d:
            predicted_probs = np.apply_along_axis(
                _binary_margin_to_prob, 1, np.vstack(tuple(predicted_probs))
            )
    elif task == "multiclass":
        predicted_probs = np.apply_along_axis(_softmax, 1, np.stack(tuple(preds_raw)))
    return predicted_probs


def predict_cls_lgbm_from_raw(
    preds_raw: np.ndarray, task: Literal["binary", "multiclass"]
) -> np.ndarray:
    """Helper function to convert raw margin predictions through a
    sigmoid to represent a probability.

    Args:
        preds_raw (ndarray):
            predictions
        lgbDataset (lightgbm.Dataset):
            dataset, containing labels, used for prediction
    Returns:
        (y_true, preds):
            tuple containg labels and predictions for further evaluation
    """
    predicted_probs = predict_proba_lgbm_from_raw(preds_raw=preds_raw, task=task)
    if task == "binary":
        pred_cls = np.array([int(p > 0.5) for p in predicted_probs])
    elif task == "multiclass":
        pred_cls = predicted_probs.argmax(axis=1)

    return pred_cls


def get_feature_shap(
    objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
    model: lgb.basic.Booster,
    X: np.ndarray,
) -> pd.DataFrame:
    """Compute model feature SHAP values and return sorted dataframe.

    Args:
        objective (str): type of the task/objective
        X (np.ndarray): dataset to compute SHAP values
        model (lgb.basic.Booster): LightGBM model

    Returns:
        feature_shap (pd.DataFrame): sorted dataframe with features and their
            SHAP values
    """
    if objective == "multiclass":
        feature_shap = np.abs(model.predict(data=X, pred_contrib=True))
        feature_shap = (
            feature_shap.reshape(
                X.shape[0], X.shape[1] + 1, model.params["num_classes"]
            )
            .sum(axis=2)
            .mean(axis=0)[:-1]
        )
    else:
        feature_shap = np.abs(model.predict(data=X, pred_contrib=True)[:, :-1]).mean(
            axis=0
        )

    feature_shap = pd.DataFrame(
        {"value": feature_shap, "feature": model.feature_name()}
    )
    feature_shap.sort_values(by="value", inplace=True)
    feature_shap.reset_index(drop=True, inplace=True)
    return feature_shap


def get_feature_importance(model: lgb.basic.Booster) -> pd.DataFrame:
    """Extract model feature importances and return sorted dataframe.

    Args:
        model (lgb.basic.Booster): LightGBM model

    Returns:
        feature_imp (pd.DataFrame): sorted dataframe with features and their
            importances
    """
    feature_imp = pd.DataFrame(
        {"value": model.feature_importance(), "feature": model.feature_name()}
    )
    feature_imp.sort_values(by="value", inplace=True)
    feature_imp.reset_index(drop=True, inplace=True)
    return feature_imp
