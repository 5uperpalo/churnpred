from typing import Tuple, Literal

import pandas as pd
import matplotlib.pyplot as plt
from eda.plotting import bar_plot
from matplotlib.figure import Figure


def correlation(
    df: pd.DataFrame,
    target: pd.Series,
    scale: Literal["log", "linear"] = "linear",
    plot: bool = False,
) -> Tuple[pd.DataFrame, Figure]:
    """Procedure to plot most correlated numerical features with
    target column.

    Args:
        df (pd.DataFrame): pandas dataframe
        target (pd.Series): target values
        scale (str): y scale of the plot
        plot (bool): whether to output the plot

    Returns:
        sorted_corr_cols (pd.DataFrame): sorted dataframe with feature and target value
            correlation
        fig (Figure): sorted bar plot with feature and target value correlation
    """
    data = df.copy()
    target_col = target.name
    correlation_df = pd.concat([data, target], axis=1).corr()
    sorted_corr_cols = correlation_df[target_col].drop(index=[target_col])
    sorted_corr_cols.sort_values(ascending=False, inplace=True)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        bar_plot(
            sorted_corr_cols,
            ax,
            title=f"Sorted features by their correlation with {target_col}",
        )
    else:
        fig = None
    ax.set_yscale(scale)
    return sorted_corr_cols, fig
