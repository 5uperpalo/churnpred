from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def drop_high_nan_cols(
    df: pd.DataFrame, threshold: float = 0.8, verbose: bool = False
) -> pd.DataFrame:
    """Returns dataframe without columns that have ratio of missingness above threshold.

    Args:
        df (pd.DataFrame): input dataframe
        threshold (float = 0.8): ratio of missingness applied per column
        verbose (bool): whether the output should be verbose
    """
    n_rows = df.shape[0]
    nan_fraction_per_col = df.apply(lambda x: x.isna().sum() / n_rows, axis=0)
    high_nan_percentage_cols = nan_fraction_per_col[
        nan_fraction_per_col > threshold
    ].index.values
    df = df.drop(high_nan_percentage_cols, axis=1)
    if verbose:
        print(
            """
            Dropped {} columns with fraction of NaN above threshold = {}.
            Affected columns: {}
            """.format(
                len(high_nan_percentage_cols), threshold, high_nan_percentage_cols
            )
        )
    return df


def drop_constant_cols(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Returns dataframe without constant columns, i.e. those with just 1 unique
    value for all rows."""
    nunique_per_col = df.apply(lambda x: x.nunique(), axis=0)
    const_cols = nunique_per_col[nunique_per_col == 1].index.values
    df = df.drop(const_cols, axis=1)
    if verbose:
        print(
            """
            Dropped {} constant columns.
            Affected columns: {}
            """.format(
                len(const_cols), const_cols
            )
        )
    return df


def drop_high_uq_cat_cols(
    df: pd.DataFrame, cat_cols: list, uq_val_count: int, verbose: bool = False
) -> pd.DataFrame:
    """Returns dataframe without categorical columns that have too many unique values

    Args:
        df (pd.DataFrame): input dataframe
        cat_cols (list): list of categorical columns
        uq_val_count (int): unique value count
        verbose (bool): whether the output should be verbose
    """
    nunique_per_col = df[cat_cols].apply(lambda x: x.nunique(), axis=0)
    high_uq_value_cat_cols = nunique_per_col[
        nunique_per_col > uq_val_count
    ].index.values
    df = df.drop(high_uq_value_cat_cols, axis=1)
    if verbose:
        print(
            """
            Dropped {} high unique value columns.
            Affected columns: {}
            """.format(
                len(high_uq_value_cat_cols), high_uq_value_cat_cols
            )
        )
    return df


def drop_highly_correlated_columns(
    df: pd.DataFrame,
    cont_cols: list,
    crosscorr_val: float = 0.95,
    verbose: bool = False,
) -> pd.DataFrame:
    """Returns dataframe without highly correlated columns, cross correlation is
    evaluated with crosscorr_val.

    Args:
        df (pd.DataFrame): input dataframe
        cont_cols (list): list of columns to evaluate correlation for
        crosscorr_val (float = 0.95): threshold value of correlation
        verbose (bool): whether the output should be verbose
    """
    corr_matrix = df[cont_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype("bool"))
    upper_above = upper.apply(lambda x: any(x > crosscorr_val), axis=1)
    to_drop = upper_above[upper_above].index.values
    df = df.drop(to_drop, axis=1)
    if verbose:
        print(
            """
            Dropped {} highly correlated columns.
            Affected columns: {}
            """.format(
                len(to_drop), to_drop
            )
        )
    return df


def nan_with_unknown_imputer(
    df: pd.DataFrame,
    columns: List[str],
    fill_token: str = "unknown",
    verbose: bool = False,
) -> pd.DataFrame:
    """Fills NAs with surrogate string, 'unknown' is default value used, it can be customized.

    Args:
        df (pd.DataFrame): input dataframe
        columns (List[str]): ist of columns that will be filled
        fillna_token (str = "unknown"): string used to replace NAs
        verbose (bool): whether the output should be verbose
    """
    df = df.copy()
    if verbose:
        sum_nan_vals = df[columns].isna().sum()
        nans_cols = []
        for c in columns:
            nans_cols.append((c, df[c].isna().sum()))
    for c in columns:
        df[c] = df[c].astype(object).fillna(fill_token)

    dfc = df.copy()
    dfc[columns] = dfc[columns].apply(lambda x: x.astype(object).fillna(fill_token))
    if verbose:
        print(
            """
            Imputed {} NaN values with {}.
            Affected columns (col, num_NaNs): {}
            """.format(
                sum_nan_vals, fill_token, nans_cols
            )
        )
    return df


def nan_with_number_imputer(
    df: pd.DataFrame,
    columns: List[str],
    fill_number: float = -1.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fills NAs with surrogate float, -1 is default value used, it can be customized.

    Args:
        df (pd.DataFrame): input dataframe
        columns (List[str]): list of columns that will be filled
        fill_number (float = -1): number used to replace NAs. Defaults to
        verbose (bool): whether the output should be verbose
    """
    dfc = df.copy()
    if verbose:
        sum_nan_vals = df[columns].isna().sum()
        nans_cols = []
        for c in columns:
            nans_cols.append((c, df[c].isna().sum()))
    dfc = df.copy()
    dfc[columns] = dfc[columns].apply(lambda x: x.astype(float).fillna(fill_number))
    if verbose:
        print(
            """
            Imputed {} NaN values with {}.
            Affected columns (col, num_NaNs): {}
            """.format(
                sum_nan_vals, fill_number, nans_cols
            )
        )
    return dfc


def nuq_in_list_col(dfs: pd.Series):
    """Returns pd.Series with the number of unique values in the lists.

    Args:
        dfs (pd.Series): input pandas series containing string list values
    """
    dfsc = dfs.copy()
    dfsc.apply(
        lambda string: (
            _nunique(list(string.replace("[", "").replace("]", "").split(", ")))
            if isinstance(string, str)
            else string
        )
    )
    return dfsc


def _nunique(List):
    """Helper method that returns the number of unique values in the list"""
    return len(set(List))


def most_frequent_in_list_col(dfs: pd.Series):
    """Returns pd.Series with the most frequent values in the lists.

    Args:
        dfs (pd.Series): input pandas series containing string list values
    """
    dfsc = dfs.copy()
    dfsc.apply(
        lambda string: (
            _most_frequent(list(string.replace("[", "").replace("]", "").split(", ")))
            if isinstance(string, str)
            else string
        )
    )
    return dfsc


def _most_frequent(List):
    """Helper method that returns the most frequent value in the list"""
    return max(set(List), key=List.count)


def replace_rare_categories_with_str_other(
    df: pd.DataFrame,
    categorical_cols: List[str],
    quantile: float = 0.05,
    surrogate_value: str = "other",
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Replaces rare category value with surrogate string.

    Args:
        df (pd.DataFrame): input dataframe
        categorical_cols (List[str]): list of columns in dataframe to process.
        quantile (float = 0.05): determines what values are considered as rare
        surrogate_value (str = "other"): string used to replace rare values

    Returns:
        Tuple[pd.DataFrame, Dict]:
            New dataframe and a dict. with mapping between orig. and surrogate values.
    """
    if surrogate_value in df[categorical_cols].values:
        raise ValueError(
            "Surrogate string - "
            + surrogate_value
            + " - is already present, choose another one."
        )

    dfc = df.copy()
    replace_dict_per_col = (
        df[categorical_cols]
        .apply(_replace_rare_dict, args=[surrogate_value, quantile])
        .to_dict()
    )
    dfc = dfc.replace(replace_dict_per_col)
    if verbose:
        sum_rare_cats = sum(
            [len(replace_dict_per_col[col]) for col in replace_dict_per_col]
        )
        rare_cats_per_col = [
            (col, list(replace_dict_per_col[col].keys()))
            for col in replace_dict_per_col
        ]
        print(
            """
            Replaced {} rare categories (val_count < {}) with {}.
            Affected columns (List[(col, rare_cats)]): {}
            """.format(
                sum_rare_cats,
                quantile,
                surrogate_value,
                rare_cats_per_col,
            )
        )
    return dfc, replace_dict_per_col


def _replace_rare_dict(df_col, surrogate_value, quantile):
    """Helper method that returns values that will be replaced by 'surrogate value'"""
    catg_counts = df_col.value_counts()
    rare_cats = catg_counts[catg_counts <= catg_counts.quantile(quantile)].index.values
    replace_dict = {k: surrogate_value for k in rare_cats}
    return replace_dict
