from typing import Any, Dict, List, Union, Literal, Optional

import pandas as pd
from lightv.general_utils import intsec
from data_preparation.base import BaseDataPreprocess
from lightv.training._base import BaseTrainer, BaseOptimizer
from lightv.training.utils import to_lgbdataset
from lightv.train_test_split._base import BaseTrainTestSplit
from lightv.training._lgb_train_function import lgb_train_function


class Trainer(BaseTrainer):
    def __init__(
        self,
        cat_cols: List[str],
        target_col: str,
        id_cols: List[str],
        objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
        groupby_cols: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        loss: Optional[Literal["focal_loss"]] = None,
        n_class: Optional[int] = None,
        optimizer: Union[Any, BaseOptimizer, None] = None,
        preprocessors: Optional[List[Union[Any, BaseDataPreprocess]]] = None,
    ):
        """Objects that governs training and parameter optimization of the lgbm model.

        Args:
            cat_cols (list): list of categorical feature column names
            target_col (str): column name that represents target
            id_cols (list): identification column names
            objective (str): type of the task/objective
            groupby_cols: List[str]: group by columns for metric computation,
                eg.user aquisition campaigns
            loss (str): type of loss function to use
                * 'None' - default for given task
                * 'focal_loss' - focal loss
            n_class (int): number of classes in the dataset
            optimizer (BaseOptimizer): parameter optimizer object
            preprocessors (List[Union[Any, BaseDataPreprocess]]):
                ordered list of objects to preprocess dataset before optimization
                and training
        """
        super(Trainer, self).__init__(
            cat_cols,
            target_col,
            id_cols,
            objective,
            groupby_cols,
            quantiles,
            n_class,
            loss,
            preprocessors,
        )
        if optimizer is not None:
            if not hasattr(optimizer, "optimize"):
                raise AttributeError(
                    "{} optimizer must have {} method".format(optimizer, "optimize")
                )
        self.optimizer = optimizer

    def train(
        self,
        df_train: pd.DataFrame,
        params: Optional[Dict] = None,
        df_valid: Optional[pd.DataFrame] = None,
    ):
        """Train the model with the parameters.

        Args:
            df_train (pd.DataFrame): training dataset
            params (dict): model paramaters
            df_valid (pd.DataFrame): optional validation dataset
        Returns:
            model (lgb.basic.Booster): trained model
        """
        if self.preprocessors:
            for prep in self.preprocessors:
                df_train_prep = prep.transform(df_train.drop(columns=[self.target_col]))
                df_valid_prep = (
                    prep.transform(df_valid.drop(columns=[self.target_col]))
                    if df_valid is not None
                    else None
                )
            if self.objective in ["binary", "multiclass"]:
                df_train_prep[self.target_col] = df_train[self.target_col].astype(int)
                if df_valid is not None:
                    df_valid_prep[self.target_col] = df_valid[self.target_col].astype(
                        int
                    )
            else:
                df_train_prep[self.target_col] = df_train[self.target_col].astype(float)
                if df_valid is not None:
                    df_valid_prep[self.target_col] = df_valid[self.target_col].astype(
                        float
                    )
        else:
            df_train_prep = df_train.copy()
            df_valid_prep = df_valid.copy() if df_valid is not None else None

        cat_cols = self.cat_cols
        if params:
            config = params
        elif (params is None) and (self.optimizer):
            config = self.optimizer.best
            if hasattr(self.optimizer, "best_to_drop"):
                df_train_prep.drop(
                    columns=self.optimizer.best_to_drop,  # type: ignore
                    inplace=True,
                )
                if df_valid is not None:
                    df_valid_prep.drop(
                        columns=self.optimizer.best_to_drop,  # type: ignore
                        inplace=True,
                    )
                cat_cols = intsec(self.cat_cols, df_train_prep.columns.values)

        elif (params is None) and (self.optimizer is None):
            config = self.base_params

        lgb_train, lgb_valid = to_lgbdataset(
            train=df_train_prep,
            cat_cols=cat_cols,
            target_col=self.target_col,
            id_cols=self.id_cols,
            valid=df_valid_prep,
        )

        if self.objective == "quantile_regression":
            self.model = {}
            for quantile in config:
                self.model[quantile] = lgb_train_function(
                    config=config[quantile],
                    lgbtrain=lgb_train,
                    lgbeval=lgb_valid,
                    feval=self.feval,
                    fobj=self.fobj,
                    with_tune=False,
                    TuneCallback_dict=None,
                )
        else:
            self.model = lgb_train_function(
                config=config,
                lgbtrain=lgb_train,
                lgbeval=lgb_valid,
                feval=self.feval,
                fobj=self.fobj,
                with_tune=False,
                TuneCallback_dict=None,
            )

    def fit(
        self,
        df: pd.DataFrame,
        splitter: Union[Any, BaseTrainTestSplit],
    ) -> pd.DataFrame:
        """Train the model and optimize the parameters.

        Args:
            df (pd.DataFrame): fitting dataset
            splitter (dict):
        Returns:
            model (lgb.basic.Booster): trained mdoel
        """
        if not hasattr(splitter, "split"):
            raise AttributeError(
                "{} splitter must have {} method".format(splitter, "split")
            )
        df_train, df_valid, df_test = splitter.split(df)

        if self.preprocessors:
            for prep in self.preprocessors:
                df_train_prep = prep.transform(df_train.drop(columns=[self.target_col]))
                df_valid_prep = prep.transform(df_valid.drop(columns=[self.target_col]))
            if self.objective in ["binary", "multiclass"]:
                df_train_prep[self.target_col] = df_train[self.target_col].astype(int)
                df_valid_prep[self.target_col] = df_valid[self.target_col].astype(int)
            else:
                df_train_prep[self.target_col] = df_train[self.target_col].astype(float)
                df_valid_prep[self.target_col] = df_valid[self.target_col].astype(float)
        else:
            df_train_prep = df_train.copy()
            df_valid_prep = df_valid.copy()

        lgb_train, lgb_valid = to_lgbdataset(
            train=df_train_prep,
            cat_cols=self.cat_cols,
            target_col=self.target_col,
            id_cols=self.id_cols,
            valid=df_valid_prep,
        )
        self.optimizer.optimize(dtrain=lgb_train, deval=lgb_valid)

        df_train_valid = pd.concat([df_train, df_valid], ignore_index=True)
        df_train_valid = df_train_valid.reset_index(drop=True)
        self.train(df_train=df_train_valid)
        metrics_dict = self.compute_metrics(df_test)
        return metrics_dict
