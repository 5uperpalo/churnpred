from pathlib import Path

from lightv import config
from lightv.general_utils import (
    diff,
    intsec,
    dill_dump,
    dill_load,
    json_dump,
    json_load,
    save_to_dir,
    set_or_create_dir,
)


def test_intsec():
    list1 = ["a", "b", "c"]
    list2 = ["b", "c"]
    assert set(["b", "c"]) == set(intsec(list1, list2))


def test_diff():
    list1 = ["a", "b", "c"]
    list2 = ["b", "c"]
    assert set(["a"]) == set(diff(list1, list2))


def test_json_load():
    content = json_load(
        file_loc=config.LOCAL_DATA_DIRNAME
        + "/"
        + config.TOY_DATASET_COLUMNS_DEFINITION_FNAME
    )
    assert type(content) == dict


# def test_json_dump():
#     json_dump(file_loc=)


# def test_dill_load():
#     content = dill_load(file_loc=)
#     assert type(content) == lgb.basic.Booster

# def test_dill_dump():
#     dill_dump(file_loc=)


def test_set_or_create_dir():
    set_or_create_dir(dirname="test")
    dir = Path("test")
    assert dir.exists()


# def test_save_to_dir():
#     save_to_dir(objects=, save_dir=)
