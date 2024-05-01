from typing import Any, Dict, List, Literal, Callable, Optional

# lightgbm default config used in optuna
# https://lightgbm.readthedocs.io/en/latest/Parameters.html
LGBM_DEFAULT_CONFIG = {
    "num_leaves": 31,
    "min_child_samples": 20,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "feature_fraction": 1.0,
}

# in BayesOpt of LightGBM some params have to be ints and not float:
LGBM_INT_PARAMS = [
    "num_iterations",
    "num_iteration",
    "n_iter",
    "num_tree",
    "num_trees",
    "num_round",
    "num_rounds",
    "nrounds",
    "num_boost_round",
    "n_estimators",
    "max_iter",
    "num_leaves",
    "num_leaf",
    "max_leaves",
    "max_leaf",
    "max_leaf_nodes",
    "max_depth",
    "max_bin",
    "max_bins",
    "bagging_freq",
    "subsample_freq",
    "min_data_in_leaf",
    "min_data_per_leaf",
    "min_data",
    "min_child_samples",
    "min_samples_leaf",
]


def set_params_to_int(params: dict) -> dict:
    """Helper function that transforms suggested float values from raytune search
    algorithm to int. LightGBM expects int for some paramaters, but some search
    algorithms, eg BayesOpt, operate only in float space.

    Args:
        params (dict): dictionary with suggested parameter values for LightGBM.

    Returns:
        params_int (dict): dictionary with suggested parameter values for LightGBM
            (proper params converted to int)
    """
    params_int = params.copy()
    for par in LGBM_INT_PARAMS:
        if par in params_int:
            params_int[par] = int(params_int[par])
    return params_int


def set_base_params(
    objective: Literal["binary", "multiclass", "regression"],
    feval: Callable,
    fobj: Callable,
    n_class: Optional[int] = None,
) -> Dict[str, Any]:
    """Set base parameters of lgbm.

    Args:
        objective (Literal["binary", "multiclass", "regression"]):
            type of task/objective
        feval (Callable): custom evaluation function
        fobj (Callable): custom objective function
        n_class (int): number of classes in the dataset

    Returns:
        params (dict): parameter dictionary for
    """
    params: Dict[str, Any] = {}

    params.update({"is_unbalance": True, "verbose": -1})

    if not fobj:
        if objective == "binary":
            params.update({"objective": "binary"})
        elif objective == "multiclass":
            params.update({"objective": "multiclass"})
        elif objective == "regression":
            params.update({"objective": "rmse"})

    if not feval:
        if objective == "binary":
            params.update({"metric": ["binary_logloss"]})
        elif objective == "multiclass":
            params.update({"metric": ["multi_logloss"]})
        elif objective == "regression":
            params.update({"metric": ["rmse"]})

    if objective == "multiclass":
        params.update({"num_classes": n_class})

    return params
