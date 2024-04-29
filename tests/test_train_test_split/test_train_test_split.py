import os
import random
import datetime

import numpy as np
import pandas as pd
import pytest
from lightv import config
from lightv.train_test_split.train_test_split import TrainTestSplit


def randomtimes(start, end, n):
    frmt = "%d-%m-%Y %H:%M:%S"
    stime = datetime.datetime.strptime(start, frmt)
    etime = datetime.datetime.strptime(end, frmt)
    td = etime - stime
    random.seed(0)
    return [random.random() * td + stime for _ in range(n)]


data = pd.DataFrame(
    {
        "TS": randomtimes("01-01-2023 00:00:00", "31-12-2023 00:00:00", 40),
        "group_id": [0] * 10 + [1] * 10 + [0] * 10 + [1] * 10,
        "feature": np.arange(0, 40),
        "target": [0] * 20 + [1] * 20,
    }
)

groupby_cols = ["group_id"]
temporal_col = "TS"


@pytest.mark.parametrize(
    "target_type, groupby_cols, temporal, temporal_col",
    [
        ("binary", None, False, None),
        ("multiclass", None, False, None),
        ("regression", None, False, None),
        ("quantile_regression", None, False, None),
        ("binary", groupby_cols, False, temporal_col),
        ("multiclass", groupby_cols, False, temporal_col),
        ("regression", groupby_cols, False, temporal_col),
        ("quantile_regression", groupby_cols, False, temporal_col),
        ("binary", groupby_cols, True, temporal_col),
        ("multiclass", groupby_cols, True, temporal_col),
        ("regression", groupby_cols, True, temporal_col),
        ("quantile_regression", groupby_cols, True, temporal_col),
    ],
)
def test_split_types(target_type, groupby_cols, temporal, temporal_col):
    train_raw, valid_raw, test_raw = TrainTestSplit(
        target_col="target",
        groupby_cols=groupby_cols,
        temporal_col=temporal_col,
        temporal=temporal,
        target_type=target_type,
    ).split(data)
    assert (len(train_raw), len(valid_raw), len(test_raw)) == (32, 4, 4)


# @pytest.mark.parametrize(
#     "save_dir",
#     [None, "temp"],
# )
# def test_split_and_save(save_dir):
#     train_raw, valid_raw, test_raw = TrainTestSplit(
#         target_col="target",
#         target_type="binary",
#         save_dir=save_dir,
#     ).split_and_save(data)
#     if save_dir:
#         train_raw_saved = pd.read_parquet(save_dir + "/train.f")
#         valid_raw_saved = pd.read_parquet(save_dir + "/valid.f")
#         test_raw_saved = pd.read_parquet(save_dir + "/test.f")
#         os.remove(save_dir + "/train.f")
#         os.remove(save_dir + "/valid.f")
#         os.remove(save_dir + "/test.f")
#         os.rmdir(save_dir)
#     else:
#         train_raw_saved = pd.read_parquet(config.LOCAL_DATA_DIRNAME + "/train.f")
#         valid_raw_saved = pd.read_parquet(config.LOCAL_DATA_DIRNAME + "/valid.f")
#         test_raw_saved = pd.read_parquet(config.LOCAL_DATA_DIRNAME + "/test.f")
#         os.remove(config.LOCAL_DATA_DIRNAME + "/train.f")
#         os.remove(config.LOCAL_DATA_DIRNAME + "/valid.f")
#         os.remove(config.LOCAL_DATA_DIRNAME + "/test.f")
#     assert all(
#         [
#             train_raw.equals(train_raw_saved),
#             valid_raw.equals(valid_raw_saved),
#             test_raw.equals(test_raw_saved),
#         ]
#     )
