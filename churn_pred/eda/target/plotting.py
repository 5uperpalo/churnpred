from typing import Literal
from itertools import zip_longest

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def prob_distrib_per_class(
    predicted_probs: np.ndarray,
    actual: np.ndarray,
    task: Literal["binary", "multiclass"],
) -> Figure:
    """Procedure to plot probability density distributions per class from LightGBM
    predictions.

    Args:
        predicted_probs (ndarray): predicted probs
        actual (ndarray): ground truth classes
        task (str): type of task

    Returns:
        fig (Figure): probability density ditributions plot per each class
    """
    if task == "binary":
        temp = pd.DataFrame({"predicted_proba": predicted_probs, "actual": actual})

        fig, ax = plt.subplots(figsize=(3, 2))
        fig.suptitle("Predicted probability density per class")
        for label, alpha, color in zip([0, 1], [1, 0.7], ["black", "red"]):
            ax.hist(
                temp[temp["actual"] == label]["predicted_proba"],
                density=True,
                bins=np.linspace(0, 1, 10),
                alpha=alpha,
                color=color,
                ec="k",
            )
        ax.set_xlim([0, 1])
        ax.set_xlabel("probability")
        ax.set_ylabel("probability_density")
    elif task == "multiclass":
        temp = pd.DataFrame(
            {
                "predicted_proba": np.take_along_axis(
                    predicted_probs, np.vstack(actual), axis=1  # type: ignore
                ).flatten(),
                "actual": actual,
            }
        )

        # Dictionary of color for each label
        color_d = dict(
            zip_longest(
                temp["actual"].unique(),
                plt.rcParams["axes.prop_cycle"].by_key()["color"],
            )
        )

        n_classes = temp["actual"].nunique()
        fig, ax = plt.subplots(
            ncols=n_classes, figsize=(n_classes * 3, 2), sharex=True, sharey=True
        )
        fig.suptitle("Predicted probability density per class")
        plt.subplots_adjust(wspace=0.3, top=0.75)

        for i, (label, gp) in enumerate(temp.groupby("actual")):
            ax[i].hist(
                gp["predicted_proba"],
                density=True,
                bins=np.linspace(0, 1, 10),
                color=color_d[label],
                ec="k",
            )
            ax[i].set_title(label)
            ax[i].set_xlim([0, 1])
            ax[i].set_xlabel("probability")
            ax[i].set_ylabel("probability_density")
    return fig


def distributions_in_binary_cls(
    df: pd.DataFrame,
    target: pd.Series,
    low_per_cut: float = 0,
    high_per_cut: float = 1,
) -> Figure:
    """Procedure to plot distributions of the features splitted
    by column split_col using workaround for violinplots in seaborn:
    * https://stackoverflow.com/a/64787568/8147433
    DISCALIMER: for now the cont_cols NA vals are filled with 0

    Args:
        df (DataFrame): pandas dataframe
        cont_cols (list): numerical features in the dataset
        target (pd.Series): target values, i.e. binary classes

    Returns:
        fig (Figure): ditribution plot per each feature
    """
    data = pd.concat([df, target], axis=1)
    cols = data.columns.values
    data["dummy"] = 0
    data = data.fillna(0).astype(float)
    fig, ax = plt.subplots(len(cols), 1, figsize=(16, len(cols) * 2))
    fig.suptitle(f"Distribution of feature values splitted by {target.name}", size=20)
    for col, i in zip(cols, range(len(cols))):
        if col != target.name:
            sns.violinplot(
                ax=ax[i],
                hue=target.name,
                x=col,
                y="dummy",
                orient="h",
                cut=0,
                showextrem=False,
                split=True,
                data=data,
            )
            ax[i].get_legend().remove()
            ax[i].set_xlim(
                data[col].quantile(low_per_cut), data[col].quantile(high_per_cut)
            )
            ax[i].get_yaxis().set_visible(False)
    plt.subplots_adjust(hspace=0.9, top=0.97)
    return fig
