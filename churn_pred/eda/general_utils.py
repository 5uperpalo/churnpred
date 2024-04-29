import math

import numpy as np


def intsec(list1: list, list2: list) -> list:
    """Simple intesection of two lists.
    Args:
        list1 (list): list1
        list2 (list): list2
    Returns:
        list (list): intersection of lists
    """
    return list(set.intersection(set(list1), set(list2)))


def entropy_calc(labels: list, base: float = np.e) -> float:
    """Computes entropy of both continuous and categorical features.
    Shamelessly stolen from :
    https://stackoverflow.com/a/45091961
    Args:
        labels (list, ndarray, Series): list of values
    Returns:
        ent (float): entropy of the list of values
    """
    n_labels = len(labels)
    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    ent = float(0)
    if n_classes > 1:
        for i in probs:
            ent -= i * math.log(i, base)
    return ent
