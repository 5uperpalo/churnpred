import warnings
from copy import deepcopy
from typing import List, Literal, Optional

import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset as lgbDataset
from optuna.integration.lightgbm import LightGBMTuner, LightGBMTunerCV

from churn_pred.training._base import BaseOptimizer

warnings.filterwarnings("ignore")


class LGBOptunaOptimizer(BaseOptimizer):
    def __init__(
        self,
        objective: Literal["binary", "multiclass", "regression"],
        n_class: Optional[int] = None,
    ):
        """Fallback/backup Optuna optimizer. Development is focused on Raytune.
        Kepping this code as backup.

        Args:
            objective (str): objective of the model
            n_class (int): number of classes in the dataset
        """
        # Optuna does not support original lighgbm implemenattion and with a workaroud it
        # is possible to get the binarry classfier with focal_loss(any custom loss)
        # running but the multiclass requires changes to Optuna code, eg. this post:
        # https://lightrun.com/answers/optuna-optuna-error-when-using-custom-metrics-in-optunaintegrationlightgbm  # noqa
        super(LGBOptunaOptimizer, self).__init__(objective, n_class)
        self.params = self.base_params

    def optimize(self, dtrain: lgbDataset, deval: lgbDataset):
        """Optimize LGBM model on provided datasets.

        Args:
            dtrain (lgbDataset): training lgb dataset
            deval (lgbDataset): evaluation lgb dataset
        """
        dtrain_copy = deepcopy(dtrain)
        deval_copy = deepcopy(deval) if deval is not None else None

        tuner = LightGBMTuner(
            params=self.params,
            train_set=dtrain_copy,
            valid_sets=deval_copy,
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(stopping_rounds=50)],
            feval=self.feval,
            # fobj=self.fobj,
        )
        tuner.run()

        self.best = tuner.best_params
        # since n_estimators is not among the params that Optuna optimizes we
        # need to add it manually. We add a high value since it will be used
        # with early_stopping_rounds
        self.best["n_estimators"] = 1000  # type: ignore
