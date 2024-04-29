from typing import Any, Dict, List, Literal, Callable, Optional

from ray import tune

# Optuna suggested values:
# https://optuna.readthedocs.io/en/stable/_modules/optuna/integration/_lightgbm_tuner/optimize.html#LightGBMTuner
_EPS = 1e-12
_DEFAULT_TUNER_TREE_DEPTH = 8
LGBM_OPTUNA_DEFAULT_CONFIG = {
    "lambda_l1": tune.uniform(1e-8, 10.0),
    "lambda_l2": tune.uniform(1e-8, 10.0),
    "num_leaves": tune.randint(2, 2**_DEFAULT_TUNER_TREE_DEPTH),
    "feature_fraction": tune.uniform(0.4, 1.0 + _EPS),
    "bagging_fraction": tune.uniform(0.4, 1.0 + _EPS),
    "bagging_freq": tune.randint(1, 7),
    "min_child_samples": tune.randint(5, 100),
}

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

# LightGBM parameter search space from Javier Rodriguez Zaurin:
# note: BayesOpt can't process "choice" and works only with floats
LGBM_CONFIG_JAVIER = {
    "learning_rate": tune.uniform(0.01, 0.3),
    "n_estimators": tune.quniform(500, 1000, 50),
    "num_leaves": tune.quniform(40, 300, 20),
    "min_child_samples": tune.quniform(20, 100, 20),
    "lambda_l1": tune.choice([0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0]),
    "lambda_l2": tune.choice([0.01, 0.05, 0.1, 0.2, 0.4, 1.0, 2.0, 4.0, 10.0]),
    # "if we drop features early in the process via running a while loop with lightgbm
    # this can dissapear and be set to 1", Javier
    "feature_fraction": tune.uniform(0.5, 1.0),
}
# see https://docs.ray.io/en/latest/_modules/ray/tune/suggest/bayesopt.html
LGBM_CONFIG_JAVIER_BAYES = LGBM_CONFIG_JAVIER.copy()
LGBM_CONFIG_JAVIER_BAYES["lambda_l1"] = tune.uniform(0.01, 10)
LGBM_CONFIG_JAVIER_BAYES["lambda_l2"] = tune.uniform(0.01, 10)
LGBM_CONFIG_JAVIER_BAYES["n_estimators"] = tune.uniform(500, 1000)
LGBM_CONFIG_JAVIER_BAYES["num_leaves"] = tune.uniform(40, 300)
LGBM_CONFIG_JAVIER_BAYES["min_child_samples"] = tune.uniform(20, 100)

# LightGBM parameter search space from:
# https://towardsdatascience.com/kagglers-guide-to-lightgbm-hyperparameter-tuning-with-optuna-in-2021-ed048d9838b5
LGBM_CONFIG_KAGGLERS_GUIDE = {
    "n_estimators": tune.uniform(500, 10000),
    "learning_rate": tune.uniform(0.001, 0.3),
    # use of this param range as is, might be "mental...", Javier :)
    "num_leaves": tune.uniform(20, 3000),
    "max_depth": tune.uniform(3, 12),
    # removed as it was making problems due to imbalance in the dataset and there were
    # no class 1 predictions
    # "min_data_in_leaf": tune.uniform(200, 10000),
    "max_bin": tune.uniform(200, 300),
    "lambda_l1": tune.uniform(0, 100),
    "lambda_l2": tune.uniform(0, 100),
    "min_gain_to_split": tune.uniform(0, 15),
    # "if we used temporal split (or in general in time dependent datasets),
    # bagging fraction should be taken with care, and probably always set to 1." Javier
    "bagging_fraction": tune.uniform(0.2, 0.95),
    "bagging_freq": tune.uniform(1, 1),
    "feature_fraction": tune.uniform(0.2, 0.95),
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
    objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
    feval: Callable,
    fobj: Callable,
    quantiles: Optional[List[float]] = None,
    quantiles_names: Optional[List[str]] = None,
    n_class: Optional[int] = None,
) -> Dict[str, Any]:
    """Set base parameters of lgbm.

    Args:
        objective (Literal["binary", "multiclass", "regression", "quantile_regression"]):
            type of task/objective
        feval (Callable): custom evaluation function
        fobj (Callable): custom objective function
        quantiles (list): list of quantiles for quantile regression
        quantiles_names (list): list of names of the quantiles
        n_class (int): number of classes in the dataset

    Returns:
        params (dict): parameter dictionary for
    """
    params: Dict[str, Any] = {}

    if objective == "quantile_regression":
        for quantile, quantile_str in zip(quantiles, quantiles_names):
            params[quantile_str] = {
                "is_unbalance": True,
                "verbose": -1,
                "alpha": quantile,
            }
    else:
        params.update({"is_unbalance": True, "verbose": -1})

    if not fobj:
        if objective == "binary":
            params.update({"objective": "binary"})
        elif objective == "multiclass":
            params.update({"objective": "multiclass"})
        elif objective == "regression":
            params.update({"objective": "rmse"})
        elif objective == "quantile_regression":
            for quantile_str in quantiles_names:
                params[quantile_str].update({"objective": "quantile"})

    if not feval:
        if objective == "binary":
            params.update({"metric": ["binary_logloss"]})
        elif objective == "multiclass":
            params.update({"metric": ["multi_logloss"]})
        elif objective == "regression":
            params.update({"metric": ["rmse"]})
        elif objective == "quantile_regression":
            for quantile_str in quantiles_names:
                params[quantile_str].update({"metric": ["quantile"]})

    if objective == "multiclass":
        params.update({"num_classes": n_class})

    return params
