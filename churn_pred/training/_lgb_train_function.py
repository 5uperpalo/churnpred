import warnings
from copy import deepcopy
from typing import Callable, Optional

import lightgbm as lgb
from lightgbm import Dataset as lgbDataset

from churn_pred.training._params import set_params_to_int


def lgb_train_function(
    config: Optional[dict],
    feval: Callable,
    lgbtrain: lgbDataset,
    lgbeval: Optional[lgbDataset] = None,
    early_stopping_rounds: int = 20,
) -> lgb.basic.Booster:
    """Training function for LighGBM using
    RayTune Weight and Biases Hyperparameter tuning.
    Args:
        config (dict): dictionary of config paramater search spaces
        feval (FunctionType): evaluation function
        fobj (FunctionType): objective function
        lgbtrain (lgbDataset): LightGBM training dataset
        lgbeval (lgbDataset): LightGBM validation dataset
        early_stopping_rounds (int): training early stopping rounds
    Returns:
        model (Booster): trained LightGBM model
    """
    lgbm_config = config.copy()
    lgbm_config = set_params_to_int(lgbm_config)
    lgbtrain_copy = deepcopy(lgbtrain)
    lgbeval_copy = deepcopy(lgbeval) if lgbeval is not None else None

    callbacks = (
        [lgb.early_stopping(stopping_rounds=early_stopping_rounds)]
        if lgbeval is not None
        else None
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        model = lgb.train(
            lgbm_config,
            lgbtrain_copy,
            valid_sets=[lgbeval_copy] if lgbeval is not None else None,
            valid_names=["val"] if lgbeval is not None else None,
            feval=feval,
            callbacks=callbacks,  # type: ignore
        )
    return model
