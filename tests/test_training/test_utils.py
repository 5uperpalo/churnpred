import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb

from churn_pred.training.utils import (
    to_lgbdataset,
    get_feature_importance,
    predict_cls_lgbm_from_raw,
    predict_proba_lgbm_from_raw,
)

# binary/regression dataset
train_df = pd.DataFrame(
    {
        "id": np.arange(0, 10),
        "cont_feature": np.arange(0, 10),
        "cont_feature_important": [0] * 5 + [1] * 5,
        "cat_feature": [0] * 5 + [1] * 5,
        "target": [0] * 5 + [1] * 5,
    },
)
valid_df = pd.DataFrame(
    {
        "id": np.arange(0, 6),
        "cont_feature": np.arange(0, 6),
        "cont_feature_important": [0] * 3 + [1] * 3,
        "cat_feature": [0] * 3 + [1] * 3,
        "target": [0] * 3 + [1] * 3,
    },
)
lgbtrain, lgbvalid = to_lgbdataset(
    train=train_df,
    cat_cols=["cat_feature"],
    target_col="target",
    id_cols=["id"],
    valid=valid_df,
)

# multiclass dataset
train_df_multi = pd.DataFrame(
    {
        "id": np.arange(0, 10),
        "cont_feature": np.arange(0, 10),
        "cont_feature_important": [0] * 3 + [1] * 3 + [2] * 4,
        "cat_feature": [0] * 5 + [1] * 5,
        "target": [0] * 3 + [1] * 3 + [2] * 4,
    },
)
valid_df_multi = pd.DataFrame(
    {
        "id": np.arange(0, 6),
        "cont_feature": np.arange(0, 6),
        "cont_feature_important": [0] * 2 + [1] * 2 + [2] * 2,
        "cat_feature": [0] * 3 + [1] * 3,
        "target": [0] * 2 + [1] * 2 + [2] * 2,
    },
)
lgbtrain_multi, lgbvalid_multi = to_lgbdataset(
    train=train_df_multi,
    cat_cols=["cat_feature"],
    target_col="target",
    id_cols=["id"],
    valid=valid_df_multi,
)

np.random.seed(0)
preds_raw_binary = np.random.uniform(low=0, high=1, size=2)
n_class = 3
preds_raw_multi = np.random.uniform(low=0, high=1, size=6).reshape(
    -1, n_class, order="F"
)


def test_to_lgbdataset_train_valid():
    lgbtrain, lgbvalid = to_lgbdataset(
        train=train_df,
        cat_cols=["cat_feature"],
        target_col="target",
        id_cols=["id"],
        valid=valid_df,
    )
    assert all(
        [
            lgbtrain.data.equals(train_df.drop(columns=["target", "id"])),
            lgbtrain.label.equals(train_df["target"]),
            lgbvalid.data.equals(valid_df.drop(columns=["target", "id"])),
            lgbvalid.label.equals(valid_df["target"]),
        ]
    )


def test_to_lgbdataset_train():
    lgbtrain, lgbvalid = to_lgbdataset(
        train=train_df,
        cat_cols=["cat_feature"],
        target_col="target",
        id_cols=["id"],
    )
    assert all(
        [
            lgbtrain.data.equals(train_df.drop(columns=["target", "id"])),
            lgbtrain.label.equals(train_df["target"]),
            lgbvalid is None,
        ]
    )


@pytest.mark.parametrize(
    ("preds_raw, task, binary2d"),
    [
        (preds_raw_binary, "binary", True),
        (preds_raw_binary, "binary", False),
        (preds_raw_multi, "multiclass", False),
    ],
)
def test_predict_proba_lgbm_from_raw(preds_raw, task, binary2d):
    preds = predict_proba_lgbm_from_raw(
        preds_raw=preds_raw,
        task=task,
        binary2d=binary2d,
    )
    if task == "binary" and binary2d is True:
        assert np.isclose(preds.sum(), 2)
    if task == "binary" and binary2d is False:
        assert preds.size == 2
    if task == "multiclass":
        assert np.isclose(preds.sum(), 2)


@pytest.mark.parametrize(
    "preds_raw, task",
    [
        (preds_raw_binary, "binary"),
        (preds_raw_multi, "multiclass"),
    ],
)
def test_predict_cls_lgbm_from_raw(preds_raw, task):
    preds = predict_cls_lgbm_from_raw(preds_raw=preds_raw, task=task)
    assert preds.size == 2
