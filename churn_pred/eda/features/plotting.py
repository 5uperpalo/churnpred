from typing import Literal

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def cross_correlation(
    df: pd.DataFrame,
    n: int = 10,
    verbose: bool = False,
) -> Figure:
    """Procedure to calculate and plot cross-correlation of features
    in the dataset.

    Args:
        df (DataFrame): pandas dataframe
        verbose (bool): show n most correlated features
        n (int): number correlated features in verbose output

    Returns:
        fig (Figure): heatmap of continuous features cross correlations
    """
    data = df.copy()

    # corrwith uses numpy which does not like pandas Float dtypes
    corr_matrix = data.astype(float).corr()
    upper_stacked_sorted = (
        corr_matrix.abs()
        .where(np.triu(np.ones(corr_matrix.shape), k=1).astype("bool"))
        .stack()
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Cross-correlation matrix", size=20)
    sns.set(font_scale=1.0)
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cbar=True,
        square=True,
        # linewidths=0.5,
        fmt=".2f",
    )
    if verbose:
        print("Top {} absolute correlations".format(n))
        print(upper_stacked_sorted[:n])
    return fig


def distributions(
    df: pd.DataFrame,
    low_per_cut: float = 0,
    high_per_cut: float = 1,
    type: Literal["box", "violin"] = "box",
) -> Figure:
    """Procedure to plot distributions of the features splitted
    by column split_col using workaround for violinplots in seaborn:
    * https://stackoverflow.com/a/64787568/8147433
    DISCALIMER: for now the cont_cols NA vals are filled with 0

    Args:
        df (DataFrame): pandas dataframe
        low_per_cut (float): lower percentile where to cut the plot for better
            readability
        high_per_cut (float): higher percentile where to cut the plot for better
            readability
        type (str): type of distribution plot

    Returns:
        fig (Figure): ditribution plot per each feature
    """
    data = df.copy()
    cols = data.columns.values
    data = data.fillna(0).astype(float)
    fig, ax = plt.subplots(len(cols), 1, figsize=(16, len(cols) * 2))
    fig.suptitle("Distribution of feature values", size=20)
    for col, i in zip(cols, range(len(cols))):
        if type == "violin":
            sns.violinplot(
                ax=ax[i],
                x=col,
                orient="h",
                # cut=0,
                # showextrem=False,
                data=data,
            )
        if type == "box":
            sns.boxplot(
                ax=ax[i],
                x=col,
                orient="h",
                data=data,
            )
        ax[i].set_xlim(
            data[col].quantile(low_per_cut), data[col].quantile(high_per_cut)
        )
        ax[i].get_yaxis().set_visible(False)
    plt.subplots_adjust(hspace=0.9)
    return fig
