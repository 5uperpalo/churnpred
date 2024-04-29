import os
from copy import copy
from typing import List, Tuple, Callable, Optional

import lightgbm as lgb

if os.environ.get("USE_MODIN") == "True":
    import modin.pandas as pd
else:
    import pandas as pd

from lightgbm import Dataset as lgbDataset
from lgb_ltv.custom_parameters import ObjectiveTypes


class SelectFeatures:
    """Selects the features to be used later in the process by LightGBM.

    The feature selection process is based on the default settings of the
    feature importance functionality in lightGBM, i.e. split importance.
    in addition there is the underlying assumption that the feature
    importance obtained when using LightGBM with its default values is
    going to be similar to that we would obatined of we run hyperparam
    optimization. It is clear that there is room for improvement in the
    design of this functionality.

    Args:
        target_col (str): target column name
        init_cat_cols (List[str]): initial categorical column names
        init_cont_cols (List[str]): initial continuous column names
        metric (Callable): Callable that will be used as a criterion to select
            the features
        reverse_score (bool): if the score is a metric, bigger is better and
            reverse_score must be set to True . If the score is a loss, lower
            is better and reverse_score must be False.
        objective (ObjectiveTypes): model obective type
        binary_threshold (Optional[float]): binary classification threshold
            in case binary classifier is used
    """

    def __init__(
        self,
        target_col: str,
        init_cat_cols: List[str],
        init_cont_cols: List[str],
        metric: Callable,
        reverse_score: bool,
        objective: ObjectiveTypes,
        binary_threshold: Optional[float] = None,
    ):
        self.target_col = target_col
        self.init_cat_cols = init_cat_cols
        self.init_cont_cols = init_cont_cols

        self.metric = metric
        self.reverse_score = reverse_score

        self.objective = objective
        self.binary_threshold = binary_threshold

    def select(
        self, dtrain: pd.DataFrame, dvalid: pd.DataFrame
    ) -> Tuple[List[str], List[str]]:
        """Main method that iteratively drops features used for training
        lighgbm model based on their importance.
        The best performing set of categorical and contunous features is
        returned."""
        score_and_cols: list = []

        selected_cat_cols = copy(self.init_cat_cols)
        selected_cont_cols = copy(self.init_cont_cols)

        sf_dtrain = dtrain.copy()
        sf_dvalid = dvalid.copy()

        while len(selected_cat_cols + selected_cont_cols) > 0:
            sf_lgbtrain = lgbDataset(
                sf_dtrain[selected_cat_cols + selected_cont_cols],
                sf_dtrain[self.target_col],
                categorical_feature=selected_cat_cols,
                free_raw_data=False,
            )
            sf_lgbvalid = lgbDataset(
                sf_dvalid[selected_cat_cols + selected_cont_cols],
                sf_dvalid[self.target_col],
                reference=sf_lgbtrain,
                free_raw_data=False,
            )

            sfmodel = lgb.train(
                {"objective": self.objective.value},
                sf_lgbtrain,
                valid_sets=[sf_lgbvalid],
                verbose_eval=False,
                callbacks=[lgb.early_stopping(stopping_rounds=30)],
            )

            preds = sfmodel.predict(sf_lgbvalid.data)

            if self.objective == ObjectiveTypes.binary:
                thresh = (
                    self.binary_threshold
                    if self.binary_threshold is not None
                    else 0.5  # noqa
                )
                score = self.metric(
                    sf_lgbvalid.label, [int(p > thresh) for p in preds]
                )  # noqa
            else:
                score = self.metric(sf_lgbvalid.label, preds)

            score_cols = [
                c
                for c in sf_lgbvalid.data.columns.tolist()
                if c != self.target_col  # noqa
            ]
            score_and_cols.append((score, score_cols))

            dfimp = pd.DataFrame(
                {
                    "fname": sfmodel.feature_name(),
                    "fimp": sfmodel.feature_importance(),
                }  # noqa
            )
            dfimp = dfimp.sort_values("fimp", ascending=False)

            drop_cols = []
            drop_cols.extend(dfimp[dfimp.fimp == 0].fname.tolist())
            drop_cols.extend(dfimp[dfimp.fimp != 0].tail(1).fname.tolist())

            sf_dtrain = sf_dtrain.drop(drop_cols, axis=1)
            sf_dvalid = sf_dvalid[sf_dtrain.columns]

            selected_cat_cols = [
                c for c in sf_dtrain.columns if c in self.init_cat_cols
            ]
            selected_cont_cols = [
                c for c in sf_dtrain.columns if c in self.init_cont_cols
            ]

        self.score_and_cols = sorted(
            score_and_cols, key=lambda x: x[0], reverse=self.reverse_score
        )

        keep_cols = self.score_and_cols[0][1]
        cat_cols_to_use = [
            c
            for c in keep_cols
            if c in self.init_cat_cols and c != self.target_col  # noqa
        ]
        cont_cols_to_use = [
            c
            for c in keep_cols
            if c in self.init_cont_cols and c != self.target_col  # noqa
        ]

        return cat_cols_to_use, cont_cols_to_use
