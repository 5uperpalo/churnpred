import numpy as np
import pandas as pd
import pytest
from lightv.training.utils import to_lgbdataset
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.suggest.bayesopt import BayesOptSearch
from lightv.training.ray_optimizer import LGBRayTuneOptimizer

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
    "opt_metric, objective, search_alg, loss, n_class, num_samples, search_params,scheduler, lgbtrain, lgbvalid",  # noqa
    [
        (
            "quantile",
            "quantile_regression",
            BayesOptSearch(),
            None,
            None,
            2,
            None,
            None,
            lgbtrain_multi,
            lgbvalid_multi,
        ),
        (
            "rmse",
            "regression",
            BayesOptSearch(),
            None,
            None,
            2,
            None,
            None,
            lgbtrain_multi,
            lgbvalid_multi,
        ),
        (
            "binary_logloss",
            "binary",
            BayesOptSearch(),
            None,
            2,
            2,
            None,
            None,
            lgbtrain_multi,
            lgbvalid_multi,
        ),
        (
            "multi_logloss",
            "multiclass",
            BayesOptSearch(),
            None,
            3,
            2,
            None,
            None,
            lgbtrain_multi,
            lgbvalid_multi,
        ),
        (
            "rmse",
            "regression",
            OptunaSearch(),
            None,
            None,
            2,
            None,
            None,
            lgbtrain_multi,
            lgbvalid_multi,
        ),
        (
            "binary_logloss",
            "binary",
            OptunaSearch(),
            None,
            2,
            2,
            None,
            None,
            lgbtrain_multi,
            lgbvalid_multi,
        ),
        (
            "multi_logloss",
            "multiclass",
            OptunaSearch(),
            None,
            3,
            2,
            None,
            None,
            lgbtrain_multi,
            lgbvalid_multi,
        ),
    ],
)
def test_drop_optimizer(
    opt_metric,
    objective,
    search_alg,
    loss,
    n_class,
    num_samples,
    search_params,
    scheduler,
    lgbtrain,
    lgbvalid,
):
    optimizer = LGBRayTuneOptimizer(
        opt_metric=opt_metric,
        objective=objective,
        quantiles=[0.25, 0.5, 0.75],
        search_alg=search_alg,
        loss=loss,
        n_class=n_class,
        num_samples=num_samples,
        search_params=search_params,
        scheduler=scheduler,
    )
    optimizer.optimize(dtrain=lgbtrain, deval=lgbvalid)
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
