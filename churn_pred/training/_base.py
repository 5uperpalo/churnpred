import warnings
from typing import Any, Dict, List, Tuple, Union, Literal, Callable, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import Dataset as lgbDataset
from sklearn.metrics import (
    f1_score,
    r2_score,
    confusion_matrix,
    mean_absolute_error,
    classification_report,
    precision_recall_curve,
)
from data_preparation.base import BaseDataPreprocess
from lightv.training.utils import (
    predict_cls_lgbm_from_raw,
    predict_proba_lgbm_from_raw,
)
from lightv.training.losses import set_feval_fobj
from lightv.training._params import set_base_params
from lightv.training.metrics import aiqc, rmse, nacil
from lightv.train_test_split._base import BaseTrainTestSplit


class Base(object):
    def __init__(
        self,
        objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
        quantiles: Optional[List[float]] = None,
        n_class: Optional[int] = None,
        loss: Optional[Literal["focal_loss"]] = None,
    ):
        """Base object with common parameters of `BaseTrainer` and `BaseOptimizer`.

        Args:
            objective (
                Literal["binary", "multiclass", "regression", "quantile_regression"]
                ): type of task/objective
            quantiles (list): list of quantiles for quantile regression
            n_class (int): number of classes in the dataset
            loss (str): type of loss function to use
                * 'None' - default for given task
                * 'focal_loss' - focal loss
        """
        self.objective = objective

        self.n_class = n_class
        if (self.objective in ["binary", "multiclass"]) and (not self.n_class):
            raise ValueError(
                "n_class must be specified for target type in ['binary', 'multiclass'])"
            )
        self.feval, self.fobj = set_feval_fobj(n_class=n_class, loss=loss)

        if quantiles:
            quantiles.sort()
            if 0.5 not in quantiles:
                warnings.warn(
                    """
                    Quantile 0.5 is not in quantiles and will be added.
                    """
                )
                quantiles.append(0.5)
        self.quantiles = quantiles
        self.quantiles.sort()

        self.base_params = set_base_params(
            objective=self.objective,
            feval=self.feval,
            fobj=self.fobj,
            quantiles=self.quantiles,
            quantiles_names=self._quantiles_to_str(),
            n_class=self.n_class,
        )

    def _quantiles_to_str(self):
        "Helper function to turn quantiles into column names in predictions"
        return ["quantile_" + str(q).replace(".", "_") for q in self.quantiles]


class BaseTrainer(Base):
    def __init__(
        self,
        cat_cols: List[str],
        target_col: str,
        id_cols: List[str],
        objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
        groupby_cols: Optional[List[str]] = None,
        quantiles: Optional[List[float]] = None,
        n_class: Optional[int] = None,
        loss: Optional[Literal["focal_loss"]] = None,
        preprocessors: Optional[List[Union[Any, BaseDataPreprocess]]] = None,
    ):
        """Object that governs optimization, training and prediction of the lgbm model.

        Args:
            cat_cols (list): list of categorical feature column names
            target_col (str): column name that represents target
            id_cols (list): identification column names
            objective (str): type of task/objective
            groupby_cols: List[str]: group by columns for metric computation,
                eg.user aquisition campaigns
            loss (str): type of loss function to use
                * 'None' - default for given task
                * 'focal_loss' - focal loss
            n_class (int): number of classes in the dataset
            preprocessors (List[Union[Any, BaseDataPreprocess]]):
                ordered list of objects to preprocess dataset before optimization
                and training
        """
        super(BaseTrainer, self).__init__(objective, quantiles, n_class, loss)

        self.cat_cols = cat_cols
        self.target_col = target_col
        self.id_cols = id_cols
        self.groupby_cols = groupby_cols
        self.model: Union[lgb.basic.Booster, dict, None] = None
        if preprocessors is not None:
            for prep in preprocessors:
                if not hasattr(prep, "transform"):
                    raise AttributeError(
                        "{} preprocessor must have {} method".format(prep, "transform")
                    )
        self.preprocessors = preprocessors

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
            model (lgb.basic.Booster): trained mdoel
        """
        raise NotImplementedError("Trainer must implement a 'train' method")

    def fit(self, df: pd.DataFrame, splitter: BaseTrainTestSplit) -> pd.DataFrame:
        """Train the model and optimize the parameters.

        Args:
            df (pd.DataFrame): fitting dataset
            splitter (dict):
        Returns:
            model (lgb.basic.Booster): trained mdoel
        """
        raise NotImplementedError("Trainer must implement a 'fit' method")

    def predict(self, df: pd.DataFrame, raw_score: bool = True) -> pd.DataFrame:
        """Predict.

        Args:
            df (pd.DataFrame): dataset
            raw_score (bool): whether to return raw output
        Returns:
            preds_raw (np.ndarray):
        """
        if self.preprocessors:
            for prep in self.preprocessors:
                df = prep.transform(df)
                if hasattr(self, "optimizer"):
                    if hasattr(self.optimizer, "best_to_drop"):  # type: ignore
                        df.drop(
                            columns=self.optimizer.best_to_drop,  # type: ignore
                            inplace=True,
                        )

        if self.objective == "quantile_regression":
            preds_raw_list = []
            for quantile in self.model:
                preds_raw_temp = self.model[quantile].predict(
                    df.drop(columns=self.id_cols), raw_score=raw_score
                )
                preds_raw_list.append(preds_raw_temp)
            preds_raw = np.asarray(preds_raw_list).transpose()
            cols_names = list(self.model)
        else:
            preds_raw = self.model.predict(  # type: ignore
                df.drop(columns=self.id_cols), raw_score=raw_score
            )
            if type(self.n_class) is int and self.n_class > 2:
                n_class_list_str = list(map(str, range(self.n_class)))
                cols_names = [self.target_col + "_" + cl for cl in n_class_list_str]
            else:
                cols_names = [self.target_col]
        return pd.DataFrame(data=preds_raw, columns=cols_names)

    def predict_proba(self, df: pd.DataFrame, binary2d: bool = False) -> pd.DataFrame:
        """Predict class probabilities.

        Args:
            df (pd.DataFrame): dataset
        Returns:
            preds_probs (np.ndarray):
        """
        if self.preprocessors:
            for prep in self.preprocessors:
                df = prep.transform(df)
                if hasattr(self, "optimizer"):
                    if hasattr(self.optimizer, "best_to_drop"):  # type: ignore
                        df.drop(
                            columns=self.optimizer.best_to_drop,  # type: ignore
                            inplace=True,
                        )

        if self.objective not in ["binary", "multiclass"]:
            raise ValueError("Can't predict class probability for regression!")

        preds_raw = self.model.predict(  # type: ignore
            df.drop(columns=self.id_cols), raw_score=True
        )
        preds_prob = predict_proba_lgbm_from_raw(
            preds_raw=preds_raw, task=self.objective, binary2d=binary2d  # type: ignore
        )
        if self.n_class > 2 or binary2d:
            n_class_list_str = list(map(str, range(self.n_class)))
            cols_names = [self.target_col + "_" + cl for cl in n_class_list_str]
        else:
            cols_names = [self.target_col]
        return pd.DataFrame(data=preds_prob, columns=cols_names)

    def predict_cls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict class.

        Args:
            df (pd.DataFrame): dataset
        Returns:
            preds_cls (np.ndarray):
        """
        if self.preprocessors:
            for prep in self.preprocessors:
                df = prep.transform(df)
                if hasattr(self, "optimizer"):
                    if hasattr(self.optimizer, "best_to_drop"):  # type: ignore
                        df.drop(
                            columns=self.optimizer.best_to_drop,  # type: ignore
                            inplace=True,
                        )

        if self.objective not in ["binary", "multiclass"]:
            raise ValueError("Can't predict class for regression!")

        preds_raw = self.model.predict(  # type: ignore
            df.drop(columns=self.id_cols), raw_score=True
        )
        preds_cls = predict_cls_lgbm_from_raw(
            preds_raw=preds_raw,
            task=self.objective,  # type: ignore
        )
        return pd.DataFrame({self.target_col: preds_cls})

    def compute_metrics(
        self,
        df: pd.DataFrame,
        with_dynamic_binary_threshold: Optional[bool] = False,
    ) -> Dict[str, Union[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Compute evaluation metrics.

        Args:
            df (pd.DataFrame): evaluation dataset
            with_dynamic_binary_threshold (bool): whether dynamic threshold will be used
                in case of binary classifier
        Returns:
            metrics_dict (dict): dictionary of computed evaluation metrics
        """
        if self.groupby_cols:
            per_group_metrics_dict = (
                df.groupby(self.groupby_cols)
                .apply(self._compute_metrics, with_dynamic_binary_threshold)
                .to_dict()
            )
            overall_metrics_dict = self._compute_metrics(
                df=df, with_dynamic_binary_threshold=with_dynamic_binary_threshold
            )
            metrics_dict = {
                "per_group": per_group_metrics_dict,
                "overall": overall_metrics_dict,
            }
        else:
            metrics_dict = self._compute_metrics(
                df=df, with_dynamic_binary_threshold=with_dynamic_binary_threshold
            )
        return metrics_dict

    def _compute_metrics(
        self,
        df: pd.DataFrame,
        with_dynamic_binary_threshold: Optional[bool] = False,
    ) -> Dict[str, Union[float, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """Helper method for compute metrics"""
        labels = df[self.target_col].values
        metrics_dict: Dict = {}

        if self.objective == "binary":
            preds_prob = self.predict_proba(
                df=df.drop(columns=[self.target_col])
            ).values
            if with_dynamic_binary_threshold:
                self.threshold_scores = self._find_binary_threshold(
                    labels,
                    preds_prob,
                )
                self.threshold = self.threshold_scores[0][0]
            else:
                self.threshold = 0.5
            preds = np.array([int(p > self.threshold) for p in preds_prob])
        elif self.objective == "multiclass":
            preds = self.predict_cls(df=df.drop(columns=[self.target_col]))
        elif self.objective in ["regression", "quantile_regression"]:
            preds = self.predict(df=df.drop(columns=[self.target_col]), raw_score=True)
            metrics_dict["sample_count"] = len(df)
            metrics_dict["mean_target_col"] = df[self.target_col].mean()

            if self.objective == "regression":
                metrics_dict["rmse"] = rmse(labels, preds)
                metrics_dict["mae"] = mean_absolute_error(labels, preds)
                metrics_dict["r2"] = r2_score(labels, preds)
            else:
                quantiles_str = self._quantiles_to_str()
                metrics_dict["rmse"] = rmse(labels, preds["quantile_0_5"])
                metrics_dict["mae"] = mean_absolute_error(labels, preds["quantile_0_5"])
                metrics_dict["r2"] = r2_score(labels, preds["quantile_0_5"])
                metrics_dict["aiqc"] = aiqc(
                    actual=labels,
                    high_quantile=preds[quantiles_str[-1]],
                    low_quantile=preds[quantiles_str[0]],
                )
                metrics_dict["nacil"] = nacil(
                    actual=labels,
                    high_quantile_predicted=preds[quantiles_str[-1]],
                    low_quantile_predicted=preds[quantiles_str[0]],
                )

        if self.objective in ["binary", "multiclass"]:
            metrics_dict["cls_report"] = classification_report(
                labels, preds, output_dict=True, zero_division=0
            )
            metrics_dict["cm"] = list(confusion_matrix(labels, preds).astype(int))
            for i in range(len(metrics_dict["cm"])):
                metrics_dict["cm"][i] = [int(a) for a in metrics_dict["cm"][i]]
            if self.objective == "binary":
                metrics_dict["prec_rec_curve"] = precision_recall_curve(labels, preds)
                metrics_dict["prec_rec_curve"] = [
                    list(arr.astype(float)) for arr in metrics_dict["prec_rec_curve"]
                ]

        return metrics_dict

    @staticmethod
    def _find_binary_threshold(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        criterion: Callable = f1_score,
    ) -> List[Tuple[float, float]]:
        """Method to find best threshold for binary classification.

        Args:
            y_true (np.ndarray): ground truth
            y_pred (np.ndarray): predicted values
            criterion (Callable): criterion to decide whitch threshold performs better
        Returns:
            threshold_score (list): list of score based on criterion
        """

        threshold_score = []
        for t in np.arange(0.2, 0.8, 0.01):
            preds_bin = [int(p > t) for p in y_pred]
            threshold_score.append((t, criterion(y_true, preds_bin)))

        return sorted(threshold_score, key=lambda x: x[1], reverse=True)


class BaseOptimizer(Base):
    def __init__(
        self,
        objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
        quantiles: Optional[List[float]] = None,
        optimize_all_quantiles: Optional[bool] = False,
        n_class: Optional[int] = None,
        loss: Optional[Literal["focal_loss"]] = None,
    ):
        """Base object to govern all tasks related to parameter optimization.

        Args:
            objective (
                Literal["binary", "multiclass", "regression", "quantile_regression"]
                ): type of task/objective
            quantiles (list): list of quantiles for quantile regression
            optimize_all_quantiles (bool): whether to optimize each quantile or just 0.5
                and copy the paramaters to other quantiles
            n_class (int): number of classes in the dataset
            loss (str): type of loss function to use
                * 'None' - default for given task
                * 'focal_loss' - focal loss
        """
        super(BaseOptimizer, self).__init__(objective, quantiles, n_class, loss)
        self.optimize_all_quantiles = optimize_all_quantiles
        self.best: Dict[str, Any] = {}  # Best hyper-parameters

    def optimize(self, dtrain: lgbDataset, deval: lgbDataset):
        """Main method to run the optimization.

        Args:
            dtrain (lgbDataset): training dataset
            deval (lgbDataset): evaluation dataset
        """
        raise NotImplementedError("Optimizer must implement a 'optimize' method")
