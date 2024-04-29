from typing import List, Tuple, Union, Literal, Optional

import pandas as pd
from lightv import config
from pandas.api.types import is_numeric_dtype, is_datetime64_dtype
from lightv.general_utils import save_to_dir
from sklearn.model_selection import train_test_split
from lightv.train_test_split._base import BaseTrainTestSplit


def assign_group_id(row, col_name):
    row_cp = row.copy()
    for nested_df in row:
        nested_df[col_name] = row.name
    return row_cp


class TrainTestSplit(BaseTrainTestSplit):
    def __init__(
        self,
        target_col: str,
        groupby_cols: Optional[List[str]] = None,
        temporal_col: Optional[str] = None,
        temporal: bool = False,
        target_type: Literal[
            "binary", "multiclass", "regression", "quantile_regression"
        ] = "binary",
        test_size: float = 0.2,
        valid_size: float = 0.5,
        random_state: Optional[int] = None,
        save_dir: Optional[str] = None,
    ):
        """
        Train/valid/test split object to split and save data depending on a target_type.

        Args:
            target_col (str): column name in dataframe that represents target
            groupby_cols: List[str]: columns to groupby before split, eg.user aquisition
                campaigns
            temporal_col (str): column used for temporal split
            temporal (bool): wether the dataset should be splitted into continuous chunks
                defined by temporal_col
            target_type (str): type of task/objective
            test_size (float): test + valid size fraction of the dataset
            valid_size (float): valid size fraction of the test + valid fraction
                of the dataset
            random_state (int): random state to keep reprodicibility of the results
            save_dir (str): saving destination directory
        """
        super(TrainTestSplit, self).__init__()

        self.target_col = target_col
        # consider automatic target type detection
        self.target_type = target_type
        self.test_size = test_size
        self.valid_size = valid_size
        self.random_state = random_state
        self.save_dir = save_dir
        self.temporal = temporal
        self.temporal_col = temporal_col
        self.groupby_cols = groupby_cols

    def split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Aggregator train/valid/test split method for _split_temporal, _split_cls
        and _split_regression methods, based on target_type.

        Args:
            df (pd.DataFrame): dataset

        Returns:
            train (pd.DataFrame): train dataset
            valid (pd.DataFrame): valid dataset
            test (pd.DataFrame): test dataset
        """
        if self.groupby_cols:
            # creates train, valid, test per group -> concatenates into series which has
            # index=group_id and each row had list value of [train, valid, test] ->
            # temp_df_nested with train, valid, test columns ->
            # add group_id to each nested dataframe -> concatenate all dataframes in
            # each column
            temp = df.groupby(self.groupby_cols).apply(self._split)
            temp_df_nested = pd.DataFrame(temp.tolist(), index=temp.index)
            temp_df_nested_w_ids = temp_df_nested.apply(
                assign_group_id, col_name=temp_df_nested.index.name, axis=1
            )
            train = pd.concat(temp_df_nested_w_ids.iloc[:][0].to_list())
            valid = pd.concat(temp_df_nested_w_ids.iloc[:][1].to_list())
            test = pd.concat(temp_df_nested_w_ids.iloc[:][2].to_list())
        else:
            train, valid, test = self._split(df=df)

        train = train.reset_index(drop=True)
        valid = valid.reset_index(drop=True)
        test = test.reset_index(drop=True)

        return train, valid, test

    def _split(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Helper method for `self.split()`."""
        if df.shape[0] < int(1 / self.test_size):
            # TODO check also self.valid_size ?
            train = df.copy()
            valid = pd.DataFrame()
            test = pd.DataFrame()
        elif self.temporal:
            train, valid, test = self._split_temporal(df)
        elif self.target_type in ["binary", "multiclass"]:
            train, valid, test = self._split_cls(df)

        elif self.target_type in ["regression", "quantile_regression"]:
            train, valid, test = self._split_regression(df)

        return train, valid, test

    def split_and_save(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Aggregator method for train/valid/test split and saveing the datasets.

        Args:
            df (pd.DataFrame): dataset

        Returns:
            train (pd.DataFrame): train dataset
            valid (pd.DataFrame): valid dataset
            test (pd.DataFrame): test dataset
        """
        train, valid, test = self.split(df)

        save_dir = config.LOCAL_DATA_DIRNAME if self.save_dir is None else self.save_dir

        save_to_dir(
            objects=[(train, "train.f"), (valid, "valid.f"), (test, "test.f")],
            save_dir=save_dir,
        )

        return train, valid, test

    def _split_temporal(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Tempora train/valid/test split based on column with temporal values.
        Works with both int and datetime columns.

        Args:
            df (pd.DataFrame): dataset

        Returns:
            train (pd.DataFrame): train dataset
            valid (pd.DataFrame): valid dataset
            test (pd.DataFrame): test dataset
        """
        dfc = df.copy()
        len_dfc = len(dfc)
        if self.temporal_col and (
            is_datetime64_dtype(dfc[self.temporal_col])
            or is_numeric_dtype(dfc[self.temporal_col])
        ):
            dfc = dfc.sort_values(by=[self.temporal_col], ignore_index=True)
        else:
            dfc = dfc.sort_index()
        train = dfc.iloc[: int(len_dfc * (1 - self.test_size)), :]
        valid = dfc.iloc[
            int(len_dfc * (1 - self.test_size)) : int(
                len_dfc * (1 - self.test_size + (self.test_size * self.valid_size))
            ),
            :,
        ]
        test = dfc.iloc[
            int(len_dfc * (1 - self.test_size + (self.test_size * self.valid_size))) :,
            :,
        ]
        return train, valid, test

    def _split_cls(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train/valid/test split based on column with class labels.

        Args:
            df (pd.DataFrame): dataset

        Returns:
            train (pd.DataFrame): train dataset
            valid (pd.DataFrame): valid dataset
            test (pd.DataFrame): test dataset
        """
        train, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[self.target_col],
        )
        valid, test = train_test_split(
            test,
            test_size=self.valid_size,
            random_state=self.random_state,
            stratify=test[self.target_col],
        )

        return train, valid, test

    def _split_regression(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Basic random train/valid/test split.

        Args:
            df (pd.DataFrame): dataset

        Returns:
            train (pd.DataFrame): train dataset
            valid (pd.DataFrame): valid dataset
            test (pd.DataFrame): test dataset
        """
        train, test = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        valid, test = train_test_split(
            test,
            test_size=self.valid_size,
            random_state=self.random_state,
        )

        return train, valid, test
