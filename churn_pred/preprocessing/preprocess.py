import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from churn_pred.utils import intsec
from churn_pred.preprocessing.label_encoder import LabelEncoder
from churn_pred.preprocessing.preprocess_data import (
    drop_constant_cols,
    drop_high_nan_cols,
    nan_with_number_imputer,
    nan_with_unknown_imputer,
    drop_highly_correlated_columns,
)


class PreprocessData:
    """Object to preprocess the dataset.
    Args:
        target_col (str): target column name
        id_cols (List[str]): id columns
        cat_cols (Optional[List[str]]): list of categorical column names
        cont_cols (Optional[List[str]]): list of continuous column names
    """

    def __init__(
        self,
        target_col: str,
        id_cols: str,
        cat_cols: Optional[List[str]] = None,
        cont_cols: Optional[List[str]] = None,
    ):
        self.target_col = target_col
        self.id_cols = id_cols
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols

        self.is_fitted = False

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit peprocessor and transform dataset in training step."""

        dfc = df.drop(columns=self.id_cols).copy()

        dfc = dfc.pipe(drop_constant_cols)

        self.cat_cols = intsec(list(dfc), self.cat_cols)

        if self.cat_cols is None:
            self.cat_cols = self._infere_cat_cols(dfc)
            dfc_cat_cols = drop_high_nan_cols(dfc[self.cat_cols])
            dfc = dfc.drop(columns=self.cat_cols)
            dfc = pd.concat([dfc, dfc_cat_cols], axis=1)
            self.cat_cols = dfc_cat_cols.columns.values.tolist()

        if self.cont_cols is None:
            self.init_cont_cols = self._infere_cont_cols(dfc, self.cat_cols)
            dfc_init_cont_cols = drop_high_nan_cols(dfc[self.init_cont_cols])
            dfc = dfc.drop(columns=self.init_cont_cols)
            dfc = pd.concat([dfc, dfc_init_cont_cols], axis=1)
            self.init_cont_cols = dfc_init_cont_cols.columns.values.tolist()

            dfc = nan_with_number_imputer(dfc, self.init_cont_cols, -9999.0)

            (
                dfc,
                self.final_cont_cols,
            ) = self._drop_highly_corr_cols_and_get_final_cont_cols(
                dfc, self.init_cont_cols
            )

        else:
            self.final_cont_cols = list(
                set(self.cont_cols).intersection(set(dfc.columns.values))
            )

            dfc = nan_with_number_imputer(dfc, self.final_cont_cols, -9999.0)

        dfc = nan_with_unknown_imputer(dfc, self.cat_cols)

        dfc = dfc[self.cat_cols + self.final_cont_cols + [self.target_col]]

        self.label_encoder = LabelEncoder(self.cat_cols)
        dfc_le = self.label_encoder.fit_transform(dfc)

        dfc_le = self._change_int_float_types(dfc_le)

        dfc_le[self.id_cols] = df[self.id_cols]

        self.is_fitted = True

        return dfc_le

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataset in inference step."""

        if not self.is_fitted:
            raise NotFittedError(
                """This instance of 'PreprocessData' has not been fitted yet.
                Please, run 'fit' first"""
            )

        dfc = df.drop(columns=self.id_cols).copy()

        try:
            dfc = dfc[self.cat_cols + self.final_cont_cols + [self.target_col]]
        except KeyError:
            dfc = dfc[self.cat_cols + self.final_cont_cols]

        # dfc = dfc.replace(self.replace_rare_categories_dict)

        dfc = nan_with_unknown_imputer(dfc, self.cat_cols)

        dfc = nan_with_number_imputer(
            dfc,
            list(set(self.final_cont_cols)),
            -9999.0,  # noqa
        )

        dfc_le = self.label_encoder.transform(dfc)

        dfc_le = self._change_int_float_types(dfc_le)

        dfc_le[self.id_cols] = df[self.id_cols]

        return dfc_le

    def fit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Just to keep familiar naming convention with sklearn."""
        return self.fit_transform(df)

    def _drop_highly_corr_cols_and_get_final_cont_cols(
        self, df: pd.DataFrame, cont_cols: List[str]
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Drop highly correlated columns to decrease the size of the dataset
        by dropping redundant information."""

        df = drop_highly_correlated_columns(df, cont_cols)

        final_cont_cols = [
            c for c in df.columns if c not in self.cat_cols + [self.target_col]
        ]

        return df, final_cont_cols

    def _infere_cat_cols(self, df: pd.DataFrame):
        """Guess the categorical columns by excluding clearly continuous
        columns - int, float"""

        cat_cols = []
        for col in df.columns:
            if (
                df[col].dtype not in ["int32", "int64"]
                and df[col].dtype not in ["float32", "float64"]
                and col not in [self.target_col]
            ):
                cat_cols.append(col)
        return cat_cols

    def _infere_cont_cols(
        self, df: pd.DataFrame, cat_cols: Optional[List[str]]
    ):  # noqa
        """Guess the continuous columns by excluding clearly categorical,
        and target_col and including only int or float."""

        if cat_cols is not None:
            cont_cols = [c for c in df.columns if c not in cat_cols + [self.target_col]]
        else:
            cont_cols = []
            for col in df.columns:
                if (df[col].dtype == "int" or df[col].dtype == "float") and col not in [
                    self.target_col
                ]:
                    cont_cols.append(col)

        return cont_cols

    def _change_int_float_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Change int to float as int causes data type issues in some ml
        methods, eg. lightgbm."""
        dfc = df.copy()
        dfc = dfc.astype(
            dict.fromkeys(dfc.select_dtypes(np.int64).columns, np.int32)
        )  # noqa
        dfc = dfc.astype(
            dict.fromkeys(dfc.select_dtypes(np.float64).columns, np.float32)
        )

        return dfc
