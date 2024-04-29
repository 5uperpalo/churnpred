import numpy as np
import pandas as pd
import pytest
from lightv.training.utils import to_lgbdataset
from lightv.training.optuna_optimizer import LGBOptunaOptimizer

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


@pytest.mark.parametrize(
    "objective, n_class, dtrain, deval",
    [
        ("quantile_regression", None, lgbtrain, lgbvalid),
        ("regression", None, lgbtrain, lgbvalid),
        ("binary", 2, lgbtrain, lgbvalid),
        ("multiclass", 3, lgbtrain_multi, lgbvalid_multi),
    ],
)
def test_optuna_optimizer(objective, n_class, dtrain, deval):
    optimizer = LGBOptunaOptimizer(
        objective=objective,
        quantiles=[0.25, 0.5, 0.75],
        n_class=n_class,
    )
    optimizer.optimize(dtrain=dtrain, deval=deval)
    if objective == "quantile_regression":
        assert all(
            [
                bool(optimizer.best),
                type(optimizer.best) == dict,
                type(optimizer.best["quantile_0_25"]) == dict,
                type(optimizer.best["quantile_0_5"]) == dict,
                type(optimizer.best["quantile_0_75"]) == dict,
            ]
        )
    else:
        assert all([bool(optimizer.best), type(optimizer.best) == dict])
