from copy import copy
from typing import List, Literal, Optional

import shap
import numpy as np
import lightgbm as lgb
from shap.plots._decision import DecisionPlotResult

from churn_pred.explain._base import BaseExplainer
from churn_pred.training.utils import predict_proba_lgbm_from_raw


class ShapExplainer(BaseExplainer):
    def __init__(
        self,
    ):
        super(ShapExplainer, self).__init__()

    def fit(
        self,
        model: lgb.basic.Booster,
        objective: Literal["binary"] = "binary",
        explainer_type: Literal["kernel", "tree", "default"] = "default",
        background_sample_count: Optional[int] = 100,
        X_train: Optional[np.ndarray] = None,
    ) -> None:
        """Fit SHAP explainer.

        Args:
            model (lgb.basic.Booster): trained model
            objective (str): type of the task/objective
            explainer_type (str): type of SHAP explainer
            background_sample_count (int): 'The background dataset to use for integrating
                out features', see: https://shap-lrjball.readthedocs.io/en/latest/generated/shap.KernelExplainer.html  # noqa
            X_train (lgb.basic.Booster): training dataset for kernel explainer
        Returns:
            self.model (): model in a form integratable with SHAP explainer
            self.explainer (): fitted SHAP explainer
        """
        self.explainer_type = explainer_type
        self.objective = objective
        self.model = model

        if self.explainer_type == "kernel":
            self.explainer = shap.KernelExplainer(
                model=self.model.predict,
                data=X_train[
                    np.random.choice(
                        X_train.shape[0], size=background_sample_count, replace=False
                    )
                ],
                model_output="raw",
            )
        elif self.explainer_type == "tree":
            self.explainer = shap.TreeExplainer(
                model=self.model,
                model_output="raw",
            )
        elif self.explainer_type == "default":
            pass

    def explain_decision_plot(
        self, X_tab_explain: np.ndarray, feature_names: Optional[list] = None
    ) -> DecisionPlotResult:
        """Process the data and pick proper decision plot based on model output and SHAP
        explainer type.

        Args:
            X_tab_explain (np.ndarray): deep tabular model component input data
        Returns:
            shap_decision_plot (): decision plot
        """
        if (self.objective in ["binary", "multiclass"]) and (
            len(X_tab_explain.shape) > 1
        ):
            raise ValueError(
                """
                Multioutput decision plots for classification support only per value
                analysis.
                """
            )
        elif len(X_tab_explain.shape) == 1:
            X_tab_explain_copy = np.expand_dims(X_tab_explain, 0)
        else:
            X_tab_explain_copy = copy(X_tab_explain)

        shap_values = self.compute_shap_values(X_tab_explain_copy)
        base_value = self.compute_base_value(X_tab_explain_copy)
        legend_labels = self._legend_labels(X_tab_explain_copy)

        if self.objective in ["binary"]:
            if (
                type(shap_values) == list or type(base_value) == list
            ) and self.objective == "binary":
                raise NotImplementedError(
                    """Binary classification with default loss is not supported. Only
                    with default loss shap explainer computes shap values for both
                    classes, see https://github.com/slundberg/shap/issues/837"""
                )
            shap_decision_plot = shap.decision_plot(
                base_value=base_value,
                shap_values=shap_values,
                features=X_tab_explain_copy,
                feature_names=feature_names,
                legend_labels=legend_labels,
                link="identity",
            )

        return shap_decision_plot

    def compute_shap_values(self, X_tab_explain: np.ndarray) -> np.ndarray:
        """Helper method to compute SHAP values for other shap plots not included in
        the ShapExplainer object.

        Args:
            X_tab_explain (np.ndarray): array of values to explain
        Returns:
            shap_values (np.ndarray): computed shap values
        """
        if self.explainer_type == "default":
            shap_values = self.model.predict(data=X_tab_explain, pred_contrib=True)[
                :, :-1
            ]
        else:
            shap_values = self.explainer.shap_values(X_tab_explain)
        return shap_values

    def compute_base_value(self, X_tab_explain: np.ndarray) -> np.ndarray:
        """Helper method to compute SHAP base/exptected value for other shap plots not included in
        the ShapExplainer object.

        Args:
            X_tab_explain (np.ndarray): array of values to explain
        Returns:
            shap_values (np.ndarray): computed shap values
        """
        if self.explainer_type == "default":
            base_value = self.model.predict(data=X_tab_explain, pred_contrib=True)[
                0, -1
            ]
        else:
            base_value = self.explainer.expected_value

        if self.objective == "multiclass":
            base_value = list(base_value)
        elif self.objective in ["binary", "regression"]:
            pass
        return base_value

    def _legend_labels(self, X_tab_explain: np.ndarray) -> List[str]:
        """Helper method to create legend with raw and predicted values.

        Args:
            X_tab_explain (np.ndarray): array of values to explain
        Returns:
            labels (list): list of labels with values
        """
        preds = np.squeeze(self.compute_preds(X_tab_explain))
        preds_raw = np.squeeze(self.model.predict(X_tab_explain))
        if self.objective == "binary":
            labels = [
                f"Sample prob. {preds.round(2):.2f} (raw {preds_raw.round(2):.2f})"
            ]
        elif self.objective == "regression":
            labels = [f"Sample val.: {preds.round(2):.2f}"]
        elif self.objective == "multiclass":
            labels = [
                f"""
                Class {i} prob. {preds[i].round(2):.2f} (raw {preds_raw[i].round(2):.2f})
                """
                for i in range(len(preds))
            ]
        return labels

    def compute_preds(self, X_tab_explain: np.ndarray) -> np.ndarray:
        """Helper method to compute SHAP values for other shap plots not included in
        the ShapExplainer object.

        Args:
            X_tab_explain (np.ndarray): array of values to explain
        Returns:
            preds (np.ndarray): computed shap values
        """
        preds_raw = self.model.predict(X_tab_explain)

        if self.objective in ["binary", "multiclass"]:
            preds = predict_proba_lgbm_from_raw(
                preds_raw, task=self.objective, binary2d=False  # type: ignore[arg-type]
            )
        return preds
