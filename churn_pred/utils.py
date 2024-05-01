from typing import Any, Union
from pathlib import Path

import dill


def intsec(list1: list, list2: list) -> list:
    """Simple intesection of two lists.
    Args:
        list1 (list): list1
        list2 (list): list2
    Returns:
        list (list): intersection of lists
    """
    return list(set.intersection(set(list1), set(list2)))


def dill_load(file_loc: Union[str, Path]) -> Any:
    """Helper function to open/close dill file,
    otherwise the python outputs warning that the file remains opened

    Args:
        file_loc (str): location of the file
    Returns:
        content (dict): content of dill file, usually dictionary
    """
    with open(file_loc, "rb") as f:
        content = dill.load(f)
    return content


def dill_dump(file_loc: Union[str, Path], content: object):
    """Helper function to open/close dill file and dump content into it,
    otherwise the python outputs warning that the file remains opened

    Args:
        file_loc (str): location of the file
        content (object): data that will be saved to dill, usually dictionary
    """
    with open(file_loc, "wb") as f:
        dill.dump(content, f)
