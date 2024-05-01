from pathlib import Path

from churn_pred.utils import intsec


def test_intsec():
    list1 = ["a", "b", "c"]
    list2 = ["b", "c"]
    assert set(["b", "c"]) == set(intsec(list1, list2))
