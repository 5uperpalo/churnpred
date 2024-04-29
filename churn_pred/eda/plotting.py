from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def bar_plot(df: pd.DataFrame, ax: Axes, title: str) -> Axes:
    """Helper method for unified bar plot

    Args:
        df (DataFrame): dataframe to plot
        ax (Axes): axes defining where to plot it

    Returns:
        adjusted_plot (Axes): adjusted axes
    """
    data = df.copy()
    ax.set_title(title, size=20)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("sorted features", size=12)
    return data.plot.bar(ax=ax)
