import numpy as np
import pytest
from lightv.training.losses import (
    lgbm_focal_fobj,
    lgbm_focal_loss,
    lgbm_focal_feval,
    lgbm_focal_loss_eval,
)

np.random.seed(0)
preds_raw_binary = np.random.uniform(low=0, high=1, size=2)
preds_raw_multi = np.random.uniform(low=0, high=1, size=6)
y_true = np.random.randint(low=0, high=1, size=2, dtype=int)


@pytest.mark.parametrize(
    "n_class",
    [
        (2),
        (3),
    ],
)
def test_lgbm_focal_feval(n_class):
    loss = lgbm_focal_feval(n_class=n_class)
    assert hasattr(loss, "__call__")


@pytest.mark.parametrize(
    "n_class",
    [
        (2),
        (3),
    ],
)
def test_lgbm_focal_fobj(n_class):
    loss = lgbm_focal_fobj(n_class=n_class)
    assert hasattr(loss, "__call__")


@pytest.mark.parametrize(
    "preds_raw, y_true, alpha, gamma, n_class",
    [
        (preds_raw_binary, y_true, 0.25, 1.0, 2),
        (preds_raw_multi, y_true, 0.25, 1.0, 3),
    ],
)
def test_lgbm_focal_loss(preds_raw, y_true, alpha, gamma, n_class):
    loss = lgbm_focal_loss(
        preds_raw=preds_raw, y_true=y_true, alpha=alpha, gamma=gamma, n_class=n_class
    )
    if n_class == 2:
        all(
            [
                all(np.isclose(loss[0], np.array([0.3174651, 0.3174651]))),
                all(np.isclose(loss[1], np.array([0.281275, 0.281275]))),
            ]
        )
    elif n_class == 3:
        all(
            [
                all(
                    np.isclose(
                        loss[0],
                        np.array(
                            [
                                -0.0562245,
                                -0.06024633,
                                0.44022066,
                                0.50346261,
                                0.4442637,
                                0.56833503,
                            ]
                        ),
                    )
                ),
                all(
                    np.isclose(
                        loss[1],
                        np.array(
                            [
                                0.06795953,
                                0.0710404,
                                0.29048985,
                                0.27700064,
                                0.29004577,
                                0.24880098,
                            ]
                        ),
                    )
                ),
            ]
        )


@pytest.mark.parametrize(
    "preds_raw, y_true, alpha, gamma, n_class",
    [
        (preds_raw_binary, y_true, 0.25, 1.0, 2),
        (preds_raw_multi, y_true, 0.25, 1.0, 3),
    ],
)
def test_lgbm_focal_loss_eval(preds_raw, y_true, alpha, gamma, n_class):
    loss = lgbm_focal_loss_eval(
        preds_raw=preds_raw, y_true=y_true, alpha=alpha, gamma=gamma, n_class=n_class
    )
    if n_class == 2:
        all(
            [
                (loss[0] == "focal_loss"),
                np.isclose(loss[1], 0.5192020872616736),
                loss[2] is False,
            ]
        )
    elif n_class == 3:
        all(
            [
                (loss[0] == "focal_loss"),
                np.isclose(loss[1], 0.35160565469892474),
                loss[2] is False,
            ]
        )
