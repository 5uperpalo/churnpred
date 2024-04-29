from copy import copy, deepcopy
from typing import List, Tuple, Union, Literal, Optional

from ray import tune
from lightgbm import Dataset as lgbDataset
from lightv.training import _params
from ray.tune.schedulers import AsyncHyperBandScheduler
from lightv.training._base import BaseOptimizer
from lightv.training._params import set_params_to_int
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from lightv.training._lgb_train_function import lgb_train_function
from ray.tune.schedulers.trial_scheduler import TrialScheduler


class LGBRayTuneOptimizer(BaseOptimizer):
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
    ):
        """Object to govern all tasks related to parameter optimization using RayTune.

        Args:
            opt_metric (str): metric that RayTune uses to monitor performance
                improvement/degradation
            objective (
                Literal["binary", "multiclass", "regression", "quantile_regression"]
                ):
                type of task/objective
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
        """
        super(LGBRayTuneOptimizer, self).__init__(
            objective, quantiles, optimize_all_quantiles, n_class, loss
        )

        # as in training function we use valid_names=["val"]
        self.opt_metric = "val-" + opt_metric
        self.opt_metric_mode = opt_metric_mode
        self.num_samples = num_samples
        self.search_alg = search_alg

        self.TuneCallback_dict = self._set_TuneCallback_dict()
        self.params = self._set_search_params(
            search_params=search_params,
        )

        if self.opt_metric not in list(self.TuneCallback_dict):
            raise ValueError(
                "opt_metric must be a evaluation metric in TuneCallback_dict!"
            )
        self.scheduler = scheduler

    def optimize(  # noqa: C901
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

        self.scheduler = self._set_scheduler()

        if self.objective == "quantile_regression":

            if self.optimize_all_quantiles:
                quantile_keys = list(self.params.keys())
            else:
                quantile_keys = list(["quantile_0_5"])

            for quantile in quantile_keys:
                best_config, _ = self._optimize_general(
                    dtrain=dtrain_copy, deval=deval_copy, params=self.params[quantile]
                )

                self.best[quantile] = set_params_to_int(best_config)
                if isinstance(self.search_alg, OptunaSearch):
                    self.best[quantile]["n_estimators"] = 1000  # type: ignore

            if not self.optimize_all_quantiles:
                for quantile_key, quantile in zip(self.params, self.quantiles):
                    self.best[quantile_key] = self.best["quantile_0_5"]
                    self.best[quantile_key]["alpha"] = quantile

        else:
            best_config, _ = self._optimize_general(
                dtrain=dtrain_copy, deval=deval_copy, params=self.params
            )

            self.best = best_config
            self.best = set_params_to_int(self.best)
            if isinstance(self.search_alg, OptunaSearch):
                self.best["n_estimators"] = 1000  # type: ignore

    def _optimize_general(
        self,
        dtrain: lgbDataset,
        deval: lgbDataset,
        params: dict,
    ) -> Tuple[dict, float]:
        tuner = tune.run(
            tune.with_parameters(
                lgb_train_function,
                lgbtrain=dtrain,
                lgbeval=deval,
                feval=self.feval,
                fobj=self.fobj,
                with_tune=True,
                TuneCallback_dict=self.TuneCallback_dict,
            ),
            search_alg=copy(self.search_alg),
            num_samples=self.num_samples,
            scheduler=copy(self.scheduler),
            config=params,
            metric=self.opt_metric,
            mode=self.opt_metric_mode,
            verbose=0,
        )
        return tuner.best_config, tuner.best_result[self.opt_metric]

    def _set_search_params(self, search_params: dict) -> dict:
        """Helper method to append parameter search space(for RayTune) to base_params."""
        params = copy(self.base_params)

        if not search_params:
            if isinstance(self.search_alg, BayesOptSearch):
                search_params = _params.LGBM_CONFIG_JAVIER_BAYES
            elif isinstance(self.search_alg, OptunaSearch):
                search_params = _params.LGBM_OPTUNA_DEFAULT_CONFIG
            elif self.search_alg is None:
                search_params = _params.LGBM_DEFAULT_CONFIG

        if self.objective == "quantile_regression":
            for quantile in self.base_params:
                params[quantile].update(search_params)
        else:
            params.update(search_params)
        return params

    def _set_TuneCallback_dict(self) -> dict:
        """Helper method to return callback dictionary that RayTune uses to monitor the
        tuning process in trails.


        Returns:
            TuneCallback_dict (dict): RayTune dict with callback metrics for monitoring
                of models performance
        """
        TuneCallback_dict = {}

        if self.objective in ["binary", "multiclass"]:
            if self.feval:
                TuneCallback_dict.update(
                    {
                        "val-focal_loss": "val-focal_loss",
                        "val-f1": "val-f1",
                        "val-recall": "val-recall",
                        "val-precision": "val-precision",
                        "val-accuracy": "val-accuracy",
                    }
                )
                for i in range(self.n_class):
                    TuneCallback_dict.update(
                        {
                            "val-f1_" + str(i): "val-f1_" + str(i),
                            "val-recall_" + str(i): "val-recall_" + str(i),
                            "val-precision_" + str(i): "val-precision_" + str(i),
                        }
                    )
            else:
                for metric in self.base_params["metric"]:
                    TuneCallback_dict.update({"val-" + metric: "val-" + metric})

        elif self.objective == "regression":
            for metric in self.base_params["metric"]:
                TuneCallback_dict.update({"val-" + metric: "val-" + metric})

        elif self.objective == "quantile_regression":
            for metric in self.base_params[list(self.base_params)[0]]["metric"]:
                TuneCallback_dict.update({"val-" + metric: "val-" + metric})

        return TuneCallback_dict

    def _set_scheduler(self):
        """Helper method to set default RayTune scheduler."""
        if not self.scheduler:
            self.scheduler = AsyncHyperBandScheduler(
                time_attr="training_iteration",
                max_t=100,
                grace_period=10,
                reduction_factor=3,
                brackets=1,
            )
