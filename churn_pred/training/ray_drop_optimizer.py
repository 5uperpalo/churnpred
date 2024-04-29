import warnings
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Union, Literal, Callable, Optional

import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset as lgbDataset
from lightv.training.utils import (
    to_lgbdataset,
    get_feature_shap,
    get_feature_importance,
)
from lightv.training._params import set_params_to_int
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from lightv.training.ray_optimizer import LGBRayTuneOptimizer
from lightv.training._lgb_train_function import lgb_train_function
from ray.tune.schedulers.trial_scheduler import TrialScheduler


class LGBRayTuneDropOptimizer(LGBRayTuneOptimizer):
    def __init__(
        self,
        opt_metric: str,
        objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
        quantiles: Optional[List[float]] = [0.25, 0.5, 0.75],
        optimize_all_quantiles: Optional[bool] = False,
        search_alg: Union[BayesOptSearch, OptunaSearch, None] = None,
        opt_metric_mode: Literal["min", "max"] = "min",
        loss: Optional[Literal["focal_loss"]] = None,
        n_class: Optional[int] = None,
        num_samples: int = 64,
        search_params: Optional[dict] = None,
        scheduler: Optional[TrialScheduler] = None,
        drop_feature_strategy: Literal["shap", "importance"] = "importance",
        drop_feature_stopping: Literal["threshold", "not_improving"] = "not_improving",
        drop_feature_stopping_mode: Literal["max", "min"] = "min",
        drop_feature_stopping_threshold: float = 0,
    ):
        """Object to govern all tasks related to parameter optimization using RayTune.
        Process tunes hyperparameters and incremetally drops features after each tuning
        step. Hyperparamter tuning can be 'disabled' by specifying no 'search_paramaters'
        and no 'search_alg'.

        Args:
            opt_metric (str): metric that RayTune uses to monitor performance
                improvement/degradation
            objective (
                Literal["binary", "multiclass", "regression", "quantile_regression"]
                ): type of task/objective
            quantiles (list): list of quantiles for quantile regression
            optimize_all_quantiles (bool): whether to optimize each quantile or just 0.5
                and copy the paramaters to other quantiles
            search_alg (Union[BayesOptSearch, OptunaSearch, None]): type of RayTune
                parameter search/optimization algorithm
            loss (str): type of loss function to use
                * 'None' - default for given task
                * 'focal_loss' - focal loss
            n_class (int): number of classes in the dataset
            num_samples (int): number of samples to take from parameter search space,
                ie. number of trial runs
            search_params (dict): optional RayTune search space for parameter
                optimization, if not set default values are picked from `_lgb_ray_params`
            scheduler (TrialScheduler): RayTune scheduler, if not set use predefined
                AsyncHyperBandScheduler
            drop_feature_strategy (str):
                * 'shap': drop features based on their computed SHAP values
                * 'importance': drop features based on their model importance
            drop_feature_stopping (str):
                * 'threshold': drop features until 'opt_metric' is above
                    'threshold' value
                * 'not_improving': stop dropping features if 'opt_metric' performance
                    is not improving
            drop_feature_stopping_mode (str):
                * 'max': whether the higher 'opt_metric' is preferred
                * 'min': whether the smaller 'opt_metric' is preferred
            drop_feature_stopping_threshold (float = 0): 'opt_metric' threshold
                until which(depends on 'drop_feature_stopping_not_improving') the
                feature are dropped
        """
        super(LGBRayTuneDropOptimizer, self).__init__(
            opt_metric,
            objective,
            quantiles,
            optimize_all_quantiles,
            search_alg,
            opt_metric_mode,
            loss,
            n_class,
            num_samples,
            search_params,
            scheduler,
        )

        self.drop_feature_strategy = drop_feature_strategy
        self.drop_feature_stopping = drop_feature_stopping
        self.drop_feature_stopping_mode = drop_feature_stopping_mode
        self.drop_feature_stopping_threshold = drop_feature_stopping_threshold

    def optimize(
        self,
        dtrain: lgbDataset,
        deval: lgbDataset,
    ):
        """Main method to run the optimization.

        Args:
            dtrain (lgbDataset): training dataset
            deval (lgbDataset): evaluation dataset
        """

        dtrain_copy = deepcopy(dtrain)
        deval_copy = deepcopy(deval)

        if self.objective == "quantile_regression":

            self.best_to_drop: Dict[str, Any] = {}  # type: ignore

            if self.optimize_all_quantiles:
                quantile_keys = list(self.params.keys())
            else:
                quantile_keys = list(["quantile_0_5"])
            for quantile_key in quantile_keys:
                (
                    self.best[quantile_key],
                    self.best_to_drop[quantile_key],
                ) = self._optimize_drop_features(
                    dtrain=dtrain_copy,
                    deval=deval_copy,
                    params=self.params[quantile_key],
                )

                self.best[quantile_key] = set_params_to_int(self.best[quantile_key])
                if isinstance(self.search_alg, OptunaSearch):
                    self.best[quantile_key]["n_estimators"] = 1000  # type: ignore

            if not self.optimize_all_quantiles:
                for quantile_key, quantile in zip(self.params, self.quantiles):
                    self.best[quantile_key] = self.best["quantile_0_5"]
                    self.best[quantile_key]["alpha"] = quantile
                    self.best_to_drop[quantile_key] = self.best_to_drop["quantile_0_5"]
        else:
            self.best_to_drop: List[str] = []  # type: ignore

            self.best, self.best_to_drop = self._optimize_drop_features(  # type: ignore
                dtrain=dtrain_copy,
                deval=deval_copy,
                params=self.params,
            )

            self.best = set_params_to_int(self.best)
            if isinstance(self.search_alg, OptunaSearch):
                self.best["n_estimators"] = 1000  # type: ignore

    def _optimize_drop_features(
        self,
        dtrain: lgbDataset,
        deval: lgbDataset,
        params: dict,
    ) -> Tuple[Dict, List]:
        best_config, best_metric_result = self._optimize_general(
            dtrain,
            deval,
            params,
        )
        best_metric_result = round(best_metric_result, 3)
        to_drop = self._train_drop_features_tune(
            best_config_now=best_config, lgbtrain=dtrain, lgbeval=deval
        )
        if not to_drop:
            keep_dropping = False
            best_to_drop = []
            warnings.warn(
                """
                All features have ZERO! importance, no drop tuning initiated!
                """
            )
        else:
            keep_dropping = True
            # features with 0 SHAP/importance value
            best_to_drop = to_drop[:-1]

        while keep_dropping and len(to_drop) < (dtrain.data.shape[1]):

            dtrain_dropped, deval_dropped = self._drop_features_from_data(
                to_drop=to_drop, lgbtrain=dtrain, lgbeval=deval
            )

            best_config_now, best_metric_result_now = self._optimize_general(
                dtrain_dropped,
                deval_dropped,
                params,
            )
            best_metric_result_now = round(best_metric_result_now, 3)
            if (
                (
                    (self.drop_feature_stopping == "threshold")
                    and (
                        (
                            best_metric_result_now
                            >= self.drop_feature_stopping_threshold
                            and self.drop_feature_stopping_mode == "min"
                        )
                        or (
                            best_metric_result_now
                            <= self.drop_feature_stopping_threshold
                            and self.drop_feature_stopping_mode == "max"
                        )
                    )
                )
                or (self.drop_feature_stopping == "not_improving")
                and (
                    (
                        best_metric_result_now <= best_metric_result
                        and self.drop_feature_stopping_mode == "min"
                    )
                    or (
                        best_metric_result_now >= best_metric_result
                        and self.drop_feature_stopping_mode == "max"
                    )
                )
            ):
                best_config = best_config_now
                best_metric_result = best_metric_result_now
                best_to_drop = to_drop

                to_drop_new = self._train_drop_features_tune(
                    best_config_now=best_config,
                    lgbtrain=dtrain_dropped,
                    lgbeval=deval_dropped,
                )
                if not to_drop_new:
                    keep_dropping = False
                else:
                    to_drop = to_drop + to_drop_new
            else:
                keep_dropping = False

        return best_config, best_to_drop

    def _train_drop_features_tune(
        self, best_config_now: dict, lgbtrain: lgbDataset, lgbeval: lgbDataset
    ):
        model = lgb_train_function(
            config=best_config_now,
            lgbtrain=lgbtrain,
            lgbeval=lgbeval,
            feval=self.feval,
            fobj=self.fobj,
            with_tune=False,
            TuneCallback_dict=None,
        )

        feature_imp = self._get_feature_influence(model=model, lgbtrain=lgbtrain)
        NonZeroImp_features_all = feature_imp[feature_imp["value"] > 0]
        if len(NonZeroImp_features_all.index) == 0:
            to_drop = []
        else:
            NonZeroImp_feature = NonZeroImp_features_all.index[0]
            to_drop = feature_imp.loc[
                : (NonZeroImp_feature + 1), "feature"
            ].values.tolist()
        return to_drop

    def _get_feature_influence(
        self, model: lgb.basic.Booster, lgbtrain: lgbDataset
    ) -> pd.DataFrame:
        """Helper method to get the influce of features depending if we focus on shap
        importacne values.
        """
        if self.drop_feature_strategy == "shap":
            feature_inf = get_feature_shap(
                objective=self.objective, model=model, X=lgbtrain.data
            )
        elif self.drop_feature_strategy == "importance":
            feature_inf = get_feature_importance(model=model)
        return feature_inf

    @staticmethod
    def _drop_features_from_data(to_drop, lgbtrain, lgbeval):
        lgbtrain_copy, lgbeval_copy = to_lgbdataset(
            train=pd.concat([lgbtrain.data, lgbtrain.label], axis=1).drop(
                columns=to_drop
            ),
            cat_cols=[
                col for col in lgbtrain.categorical_feature if col not in to_drop
            ],
            target_col=lgbtrain.label.name,
            valid=(
                pd.concat([lgbeval.data, lgbeval.label], axis=1).drop(columns=to_drop)
                if lgbeval is not None
                else None
            ),
        )
        return lgbtrain_copy, lgbeval_copy
