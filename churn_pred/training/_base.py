import warnings
from typing import Any, Dict, List, Tuple, Union, Literal, Callable, Optional

import numpy as np
import mlflow
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

from churn_pred.training.utils import (
    predict_cls_lgbm_from_raw,
    predict_proba_lgbm_from_raw,
)
from churn_pred.training._params import set_base_params
from churn_pred.training.metrics import aiqc, rmse, nacil
from churn_pred.preprocessing.preprocess import PreprocessData


class Base(object):
    def __init__(
        self,
        objective: Literal["binary", "multiclass", "regression"],
        n_class: Optional[int] = None,
    ):
        """Base object with common parameters of `BaseTrainer` and `BaseOptimizer`.

        Args:
            objective (
                Literal["binary", "multiclass", "regression"]
                ): type of task/objective
            n_class (int): number of classes in the dataset
        """
        self.objective = objective

        self.n_class = n_class
        if (self.objective in ["binary", "multiclass"]) and (not self.n_class):
            raise ValueError(
                "n_class must be specified for target type in ['binary', 'multiclass'])"
            )
        self.feval, self.fobj = None, None

        self.base_params = set_base_params(
            objective=self.objective,
            feval=self.feval,
            fobj=self.fobj,
            n_class=self.n_class,
        )


class BaseTrainer(Base, mlflow.pyfunc.PythonModel):
    def __init__(
        self,
        cat_cols: List[str],
        target_col: str,
        id_cols: List[str],
        objective: Literal["binary", "multiclass", "regression"],
        n_class: Optional[int] = None,
        preprocessors: Optional[List[Union[Any, PreprocessData]]] = None,
    ):
        """Object that governs optimization, training and prediction of the lgbm model.

        Args:
            cat_cols (list): list of categorical feature column names
            target_col (str): column name that represents target
            id_cols (list): identification column names
            objective (str): type of task/objective
            n_class (int): number of classes in the dataset
            preprocessors (List[Union[Any, PreprocessData]]):
                ordered list of objects to preprocess dataset before optimization
                and training
        """
        super(BaseTrainer, self).__init__(objective, n_class)

        self.cat_cols = cat_cols
        self.target_col = target_col
        self.id_cols = id_cols
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

    def fit(
        self,
        df_train: pd.DataFrame,
        df_valid: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> pd.DataFrame:
        """Train the model and optimize the parameters.

        Args:
            df_train (pd.DataFrame): training dataset
            df_valid (pd.DataFrame): validation dataset
            df_test (pd.DataFrame): testing dataset
        Returns:
            model (lgb.basic.Booster): trained model
        """
        raise NotImplementedError("Trainer must implement a 'fit' method")

    def predict(
        self, context: Optional[dict], df: pd.DataFrame, raw_score: bool = True
    ) -> pd.DataFrame:
        """Predict.

        Args:
            df (pd.DataFrame): dataset
            raw_score (bool): whether to return raw output
            context (dict): for compatibility with mlflow model methods
        Returns:
            preds_raw (np.ndarray):
        """
        # for mlflow inference service testing
        if type(df) is dict:
            df = pd.DataFrame.from_dict(df, orient="index").transpose()

        if self.preprocessors:
            for prep in self.preprocessors:
                df = prep.transform(df)
                if hasattr(self, "optimizer"):
                    if hasattr(self.optimizer, "best_to_drop"):  # type: ignore
                        df.drop(
                            columns=self.optimizer.best_to_drop,  # type: ignore
                            inplace=True,
                        )

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
            preds_raw=preds_raw,  # type: ignore
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
        elif self.objective == "regression":
            preds = self.predict(df=df.drop(columns=[self.target_col]), raw_score=True)
            metrics_dict["sample_count"] = len(df)
            metrics_dict["mean_target_col"] = df[self.target_col].mean()
            metrics_dict["rmse"] = rmse(labels, preds)
            metrics_dict["mae"] = mean_absolute_error(labels, preds)
            metrics_dict["r2"] = r2_score(labels, preds)

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
        objective: Literal["binary", "multiclass", "regression"],
        n_class: Optional[int] = None,
    ):
        """Base object to govern all tasks related to parameter optimization.

        Args:
            objective (
                Literal["binary", "multiclass", "regression"]
                ): type of task/objective
            n_class (int): number of classes in the dataset
        """
        super(BaseOptimizer, self).__init__(objective, n_class)
        self.best: Dict[str, Any] = {}  # Best hyper-parameters

    def optimize(self, dtrain: lgbDataset, deval: lgbDataset):
        """Main method to run the optimization.

        Args:
            dtrain (lgbDataset): training dataset
            deval (lgbDataset): evaluation dataset
        """
        raise NotImplementedError("Optimizer must implement a 'optimize' method")
