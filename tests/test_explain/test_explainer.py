import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb
from lightv.explain import explainer
from lightv.training.utils import to_lgbdataset

# binary/regression dataset
train_df = pd.DataFrame(
    {
        "id": np.arange(0, 50),
        "cont_feature": np.arange(0, 50),
        "cont_feature_important": [0] * 25 + [1] * 25,
        "cat_feature": [0] * 25 + [1] * 25,
        "target": [0] * 25 + [1] * 25,
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
        "id": np.arange(0, 50),
        "cont_feature": np.arange(0, 50),
        "cont_feature_important": [0] * 15 + [1] * 15 + [2] * 20,
        "cat_feature": [0] * 25 + [1] * 25,
        "target": [0] * 15 + [1] * 15 + [2] * 20,
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
    ("objective, n_class, explainer_type, lgbtrain, lgbvalid"),
    [
        ("binary", 2, "kernel", lgbtrain, lgbvalid),
        ("binary", 2, "kernel", lgbtrain, lgbvalid),
        ("binary", 2, "tree", lgbtrain, lgbvalid),
        ("binary", 2, "tree", lgbtrain, lgbvalid),
        ("multiclass", 3, "kernel", lgbtrain_multi, lgbvalid_multi),
        ("multiclass", 3, "kernel", lgbtrain_multi, lgbvalid_multi),
        ("multiclass", 3, "tree", lgbtrain_multi, lgbvalid_multi),
        ("multiclass", 3, "tree", lgbtrain_multi, lgbvalid_multi),
        ("regression", None, "kernel", lgbtrain, lgbvalid),
        ("regression", None, "tree", lgbtrain, lgbvalid),
    ],
)
def test_explainer(
    objective,
    n_class,
    explainer_type,
    lgbtrain,
    lgbvalid,
):
    if objective == "binary":
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
        }
    elif objective == "multiclass":
        params = {
            "objective": "multiclass",
            "metric": "multiclass",
            "num_classes": n_class,
            "verbose": -1,
        }
    elif objective == "regression":
        params = {
            "objective": "regression",
            "metric": "regression",
            "verbose": -1,
        }
    model = lgb.train(
        params=params,
        train_set=lgbtrain,
    )

    shap_explainer = explainer.ShapExplainer()

    shap_explainer.fit(
        model=model,
        objective=objective,
        X_train=lgbtrain.data.values,
        explainer_type=explainer_type,
        background_sample_count=50,
    )
    if objective == "binary" and explainer_type == "tree":
        with pytest.raises(NotImplementedError):
            shap_explainer.explain_decision_plot(
                X_tab_explain=lgbvalid.data.values[1],
                feature_names=list(lgbvalid.data),
            )
    else:
        shap_explainer.explain_decision_plot(
            X_tab_explain=lgbvalid.data.values[1],
            feature_names=list(lgbvalid.data),
        )
