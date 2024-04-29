from typing import Tuple, Literal, Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from churn_pred.eda.plotting import bar_plot
from churn_pred.eda.general_utils import intsec, entropy_calc


def init_check(
    df: pd.DataFrame,
    identifier: Optional[str] = None,
    cat_cols: Optional[list] = None,
    cont_cols: Optional[list] = None,
    verbose: bool = False,
) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
    """Procedure to check:
    * duplicated rows in teh dataset
    * general stats of numerical features
    * general stats of categorical features

    Args:
        df (DataFrame): pandas dataframe
        identifier (str): column which identifies unique user IDs
        cat_cols (list): categorical features in the dataset
        cont_cols (list): numerical features in the dataset

    Returns:
        duplicated_ids (int):
        cont_cols_desc (pd.DataFrame):
        cat_cols_desc (pd.DataFrame):
    """
    data = df.copy()

    if identifier:
        duplicated_ids = data[identifier].duplicated().sum()
        if verbose:
            print("[CHECK] Number of duplicated ids: {}".format(duplicated_ids))

    if cont_cols:
        cont_cols_f = intsec(data.columns.values, cont_cols)
        cont_cols_desc = data[cont_cols_f].describe().transpose()
        if verbose:
            print("[CHECK] Numerical columns")
            with pd.option_context("display.precision", 2):
                print(cont_cols_desc)

    if cat_cols:
        cat_cols_f = intsec(data.columns.values, cat_cols)
        cat_cols_desc = data[cat_cols_f].describe().transpose()
        if verbose:
            print("[CHECK] Categorical columns")
            with pd.option_context("display.precision", 2):
                print(cat_cols_desc)
    return duplicated_ids, cont_cols_desc, cat_cols_desc


def missing(
    df: pd.DataFrame,
    scale: Literal["log", "linear"] = "linear",
    plot: bool = False,
) -> Tuple[pd.DataFrame, Figure]:
    """Procedure to check fraction of missing values in the dataset.

    Args:
        df (DataFrame): pandas dataframe
        scale (str): y scale of the plot
        plot (bool): whether to output the plot

    Returns:
        missing_val_frac (DataFrame): sorted dataframe with fraction of missing values
            per feature
        fig (Figure): plot with sorted fractions of missing values in each column
    """
    data = df.copy()
    missing_val_frac = data.isna().sum() / len(data)
    missing_val_frac.sort_values(ascending=False, inplace=True)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_plot(
            missing_val_frac,
            ax,
            title="Fraction of missing values in the dataset",
        )
    else:
        fig = None
    ax.set_yscale(scale)
    return missing_val_frac, fig


def zero(
    df: pd.DataFrame,
    scale: Literal["log", "linear"] = "linear",
    plot: bool = False,
) -> Tuple[pd.DataFrame, Figure]:
    """Procedure to check fraction of zero values in the dataset.

    Args:
        df (DataFrame): pandas dataframe
        scale (str): y scale of the plot
        plot (bool): whether to output the plot

    Returns:
        zero_val_frac (DataFrame): sorted dataframe with fraction of '0' values per
            feature
        fig (Figure): plot with sorted fractions of '0' values in each column
    """
    data = df.copy()
    zero_val_frac = data.isin([0]).sum() / len(data)
    zero_val_frac.sort_values(ascending=False, inplace=True)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_plot(zero_val_frac, ax, title="Fraction of zero values in the dataset")
    else:
        fig = None
    ax.set_yscale(scale)
    return zero_val_frac, fig


def nunique(
    df: pd.DataFrame,
    scale: Literal["log", "linear"] = "linear",
    plot: bool = False,
) -> Tuple[pd.DataFrame, Figure]:
    """Procedure to plot features sorted by their number of unique values.

    Args:
        df (DataFrame): dataset
        scale (str): y scale of the plot
        plot (bool): whether to output the plot

    Returns:
        fig (Figure): plot with sorted number of unique values in features
    """
    data = df.copy()

    data_nunique = data.nunique()
    data_nunique.sort_values(ascending=True, inplace=True)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_plot(
            data_nunique.dropna(),
            ax,
            title="Features number of unique values (NA are dropped)",
        )
    else:
        fig = None
    ax.set_yscale(scale)
    return data_nunique, fig


def std(
    df: pd.DataFrame,
    scale: Literal["log", "linear"] = "linear",
    plot: bool = False,
) -> Tuple[pd.DataFrame, Figure]:
    """Procedure to plot features sorted by their variance.

    Args:
        df (DataFrame): dataset
        scale (str): y scale of the plot
        plot (bool): whether to output the plot

    Returns:
        fig (Figure): plot with sorted standard deviation of continuous features
    """
    data = df.copy()

    data_std = data.std()

    data_std.sort_values(ascending=False, inplace=True)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_plot(data_std.dropna(), ax, title="Features variance (NA are dropped)")
    else:
        fig = None
    ax.set_yscale(scale)
    return data_std, fig


def entropy(
    df: pd.DataFrame, scale: Literal["log", "linear"] = "linear", plot: bool = False
) -> Tuple[pd.DataFrame, Figure]:
    """Procedure to plot features sorted by their entropy.

    Args:
        df (DataFrame): dataset
        scale (str): y scale of the plot
        plot (bool): whether to output the plot

    Returns:
        fig (Figure): plot with sorted entropy of the features
    """
    data = df.copy()

    data = data.astype(str)
    col_entropies = data.apply(entropy_calc, axis=0)
    col_entropies.sort_values(ascending=False, inplace=True)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_plot(
            col_entropies.dropna(), ax, title="Features entropies (NA are dropped)"
        )
    else:
        fig = None
    ax.set_yscale(scale)
    return col_entropies, fig
