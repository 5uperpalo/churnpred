import warnings
from copy import deepcopy
from typing import List, Literal, Optional

import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset as lgbDataset
from lightv.training._base import BaseOptimizer
from optuna.integration.lightgbm import LightGBMTuner, LightGBMTunerCV

warnings.filterwarnings("ignore")


class LGBOptunaOptimizer(BaseOptimizer):
    def __init__(
        self,
        objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
        quantiles: Optional[List[float]] = [0.25, 0.5, 0.75],
        optimize_all_quantiles: Optional[bool] = False,
        n_class: Optional[int] = None,
    ):
        """Fallback/backup Optuna optimizer. Development is focused on Raytune.
        Kepping this code as backup.

        Args:
            objective (str): objective of the model
            quantiles (list): list of quantiles for quantile regression
            n_class (int): number of classes in the dataset
            quantile (list): quantile list for the case of 'quantile_regression'
        """
        # Optuna does not support original lighgbm implemenattion and with a workaroud it
        # is possible to get the binarry classfier with focal_loss(any custom loss)
        # running but the multiclass requires changes to Optuna code, eg. this post:
        # https://lightrun.com/answers/optuna-optuna-error-when-using-custom-metrics-in-optunaintegrationlightgbm  # noqa
        loss = None
        super(LGBOptunaOptimizer, self).__init__(
            objective, quantiles, optimize_all_quantiles, n_class, loss
        )
        self.params = self.base_params

    def optimize(self, dtrain: lgbDataset, deval: lgbDataset):
        """Optimize LGBM model on provided datasets.

        Args:
            dtrain (lgbDataset): training lgb dataset
            deval (lgbDataset): evaluation lgb dataset
        """
        dtrain_copy = deepcopy(dtrain)
        deval_copy = deepcopy(deval) if deval is not None else None

        if self.objective == "quantile_regression":

            if self.optimize_all_quantiles:
                quantile_keys = list(self.params.keys())
            else:
                quantile_keys = list(["quantile_0_5"])

            for quantile_key in quantile_keys:
                tuner = LightGBMTuner(
                    params=self.params[quantile_key],
                    train_set=dtrain_copy,
                    valid_sets=deval_copy,
                    num_boost_round=1000,
                    verbose_eval=False,
                    callbacks=[lgb.early_stopping(stopping_rounds=50)],
                    feval=self.feval,
                    fobj=self.fobj,
                )
                tuner.run()

                self.best[quantile_key] = tuner.best_params
                # since n_estimators is not among the params that Optuna optimizes we
                # need to add it manually. We add a high value since it will be used
                # with early_stopping_rounds
                self.best[quantile_key]["n_estimators"] = 1000  # type: ignore

            if not self.optimize_all_quantiles:
                for quantile_key, quantile in zip(self.params, self.quantiles):
                    self.best[quantile_key] = self.best["quantile_0_5"]
                    self.best[quantile_key]["alpha"] = quantile
        else:
            tuner = LightGBMTuner(
                params=self.params,
                train_set=dtrain_copy,
                valid_sets=deval_copy,
                num_boost_round=1000,
                verbose_eval=False,
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                feval=self.feval,
                fobj=self.fobj,
            )
            tuner.run()

            self.best = tuner.best_params
            # since n_estimators is not among the params that Optuna optimizes we
            # need to add it manually. We add a high value since it will be used
            # with early_stopping_rounds
            self.best["n_estimators"] = 1000  # type: ignore
