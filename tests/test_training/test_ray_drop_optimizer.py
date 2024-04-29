import pandas as pd
import pytest
from sklearn.datasets import make_classification
from lightv.training.utils import to_lgbdataset
from ray.tune.suggest.bayesopt import BayesOptSearch
from lightv.training.ray_drop_optimizer import LGBRayTuneDropOptimizer

X, y = make_classification(
    n_samples=110,
    n_features=15,
    n_informative=10,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    random_state=0,
    shuffle=False,
)
train_df = pd.concat(
    [
        pd.DataFrame(data=X[:100], columns=list(map(str, range(15)))),
        pd.DataFrame({"target": y[:100]}),
    ],
    axis=1,
)
valid_df = pd.concat(
    [
        pd.DataFrame(data=X[100:], columns=list(map(str, range(15)))),
        pd.DataFrame({"target": y[100:]}),
    ],
    axis=1,
)
lgbtrain, lgbvalid = to_lgbdataset(
    train=train_df,
    cat_cols=[],
    target_col="target",
    id_cols=[],
    valid=valid_df,
)


@pytest.mark.parametrize(
    "opt_metric, objective, search_alg, loss, n_class, num_samples, search_params, scheduler, drop_feature_strategy, drop_feature_stopping, lgbtrain, lgbvalid",  # noqa
    [
        (
            "quantile",
            "quantile_regression",
            BayesOptSearch(),
            None,
            2,
            2,
            None,
            None,
            "importance",
            "not_improving",
            lgbtrain,
            lgbvalid,
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
            "importance",
            "not_improving",
            lgbtrain,
            lgbvalid,
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
            "shap",
            "not_improving",
            lgbtrain,
            lgbvalid,
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
            "importance",
            "threshold",
            lgbtrain,
            lgbvalid,
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
            "shap",
            "threshold",
            lgbtrain,
            lgbvalid,
        ),
        (
            "binary_logloss",
            "binary",
            None,
            None,
            2,
            2,
            None,
            None,
            "shap",
            "threshold",
            lgbtrain,
            lgbvalid,
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
    drop_feature_strategy,
    drop_feature_stopping,
    lgbtrain,
    lgbvalid,
):
    optimizer = LGBRayTuneDropOptimizer(
        opt_metric=opt_metric,
        objective=objective,
        quantiles=[0.25, 0.5, 0.75],
        search_alg=search_alg,
        loss=loss,
        n_class=n_class,
        num_samples=num_samples,
        search_params=search_params,
        scheduler=scheduler,
        drop_feature_strategy=drop_feature_strategy,
        drop_feature_stopping=drop_feature_stopping,
    )
    optimizer.optimize(dtrain=lgbtrain, deval=lgbvalid)
    if objective == "quantile_regression":
        assert all(
            [
                bool(optimizer.best),
                bool(optimizer.best_to_drop),
                type(optimizer.best) == dict,
                type(optimizer.best["quantile_0_25"]) == dict,
                type(optimizer.best["quantile_0_5"]) == dict,
                type(optimizer.best["quantile_0_75"]) == dict,
                type(optimizer.best_to_drop) == dict,
                type(optimizer.best_to_drop["quantile_0_25"]) == list,
                type(optimizer.best_to_drop["quantile_0_5"]) == list,
                type(optimizer.best_to_drop["quantile_0_75"]) == list,
            ]
        )
    else:
        assert all(
            [
                bool(optimizer.best),
                bool(optimizer.best_to_drop),
                type(optimizer.best) == dict,
                type(optimizer.best_to_drop) == list,
            ]
        )
