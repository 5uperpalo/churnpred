from typing import List, Tuple, Literal, Optional

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb
from lightgbm import Dataset as lgbDataset
from lightv.training._base import BaseOptimizer
from lightv.training.trainer import Trainer
from sklearn.model_selection import train_test_split

# binary/regression dataset
train_df = pd.DataFrame(
    {
        "id": np.arange(0, 100),
        "group_id": [0] * 80 + [1] * 20,
        "cont_feature": np.arange(0, 100),
        "cont_feature_important": [0] * 50 + [1] * 50,
        "cat_feature": [0] * 50 + [1] * 50,
        "target": [0] * 50 + [1] * 50,
    },
)
valid_df = pd.DataFrame(
    {
        "id": np.arange(0, 6),
        "group_id": [0] * 4 + [1] * 2,
        "cont_feature": np.arange(0, 6),
        "cont_feature_important": [0] * 3 + [1] * 3,
        "cat_feature": [0] * 3 + [1] * 3,
        "target": [0] * 3 + [1] * 3,
    },
)

# multiclass dataset
train_df_multi = pd.DataFrame(
    {
        "id": np.arange(0, 100),
        "group_id": [0] * 80 + [1] * 20,
        "cont_feature": np.arange(0, 100),
        "cont_feature_important": [0] * 30 + [1] * 30 + [2] * 40,
        "cat_feature": [0] * 50 + [1] * 50,
        "target": [0] * 30 + [1] * 30 + [2] * 40,
    },
)
valid_df_multi = pd.DataFrame(
    {
        "id": np.arange(0, 6),
        "group_id": [0] * 4 + [1] * 2,
        "cont_feature": np.arange(0, 6),
        "cont_feature_important": [0] * 2 + [1] * 2 + [2] * 2,
        "cat_feature": [0] * 3 + [1] * 3,
        "target": [0] * 2 + [1] * 2 + [2] * 2,
    },
)

groupby_cols = ["group_id"]


class TestOptimizer(BaseOptimizer):
    def __init__(
        self,
        objective: Literal["binary", "multiclass", "regression", "quantile_regression"],
        quantiles: Optional[List[float]] = [0.25, 0.5, 0.75],
        optimize_all_quantiles: Optional[bool] = False,
        n_class: Optional[int] = None,
        loss: Optional[Literal["focal_loss"]] = None,
    ):
        super(TestOptimizer, self).__init__(
            objective, quantiles, optimize_all_quantiles, n_class, loss
        )

    def optimize(self, dtrain: lgbDataset, deval: lgbDataset):
        self.best = self.base_params


class TestPreprocessor:
    def fit_transform(self, df: pd.DataFrame):
        return df

    def transform(self, df: pd.DataFrame):
        return df


class TestSplitter:
    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        train, test = train_test_split(
            df,
            test_size=0.2,
            random_state=0,
        )
        valid, test = train_test_split(
            test,
            test_size=0.5,
            random_state=0,
        )

        return train, valid, test


test_preprocessors = [TestPreprocessor()]


@pytest.mark.parametrize(
    "cat_cols, target_col, id_cols, objective, groupby_cols, loss, n_class, preprocessors, train_df, valid_df",  # noqa
    [
        (
            ["cat_feature"],
            "target",
            ["id"],
            "quantile_regression",
            groupby_cols,
            None,
            None,
            None,
            train_df,
            valid_df,
        ),
        (
            ["cat_feature"],
            "target",
            ["id"],
            "quantile_regression",
            None,
            None,
            None,
            None,
            train_df,
            valid_df,
        ),
        (
            ["cat_feature"],
            "target",
            ["id"],
            "regression",
            None,
            None,
            None,
            None,
            train_df,
            valid_df,
        ),
        (
            ["cat_feature"],
            "target",
            ["id"],
            "binary",
            None,
            None,
            2,
            None,
            train_df,
            valid_df,
        ),
        (
            ["cat_feature"],
            "target",
            ["id"],
            "multiclass",
            None,
            None,
            3,
            None,
            train_df_multi,
            valid_df_multi,
        ),
        (
            ["cat_feature"],
            "target",
            ["id"],
            "binary",
            None,
            "focal_loss",
            2,
            None,
            train_df,
            valid_df,
        ),
        (
            ["cat_feature"],
            "target",
            ["id"],
            "multiclass",
            None,
            "focal_loss",
            3,
            None,
            train_df_multi,
            valid_df_multi,
        ),
    ],
)
def test_trainer_fit_train(
    cat_cols,
    target_col,
    id_cols,
    objective,
    groupby_cols,
    loss,
    n_class,
    preprocessors,
    train_df,
    valid_df,
):
    test_optimizer = TestOptimizer(
        objective=objective, n_class=n_class, quantiles=[0.25, 0.5, 0.75], loss=loss
    )
    trainer = Trainer(
        cat_cols=cat_cols,
        target_col=target_col,
        id_cols=id_cols,
        objective=objective,
        groupby_cols=groupby_cols,
        quantiles=[0.25, 0.75],
        loss=loss,
        n_class=n_class,
        optimizer=test_optimizer,
        preprocessors=preprocessors,
    )
    metrics_dict = trainer.fit(df=train_df, splitter=TestSplitter())
    trainer.train(
        df_train=train_df,
        params=None,
        df_valid=valid_df,
    )
    if objective == "quantile_regression":
        assert all(
            [
                (type(metrics_dict) == dict),
                (type(trainer.model) == dict),
                (type(trainer.model["quantile_0_25"]) == lgb.basic.Booster),
                (type(trainer.model["quantile_0_5"]) == lgb.basic.Booster),
                (type(trainer.model["quantile_0_75"]) == lgb.basic.Booster),
            ]
        )
    else:
        assert all(
            [
                (type(metrics_dict) == dict),
                (type(trainer.model) == lgb.basic.Booster),
            ]
        )
