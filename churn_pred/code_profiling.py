import json
import logging
from time import time
from functools import wraps

import numpy as np
from memory_profiler import memory_usage


def timer(func):
    """Wrapper to time and log the function execution.
    Parameters:
        func: function
    """

    @wraps(func)
    def timer_func(*args, **kwargs):
        start_time = time()
        value = func(*args, **kwargs)
        end_time = time()
        logging.info(
            f"Finished {func.__name__} in {(end_time - start_time):.4f} seconds."  # noqa
        )
        return value

    return timer_func


def ram_usage(func):
    """Wrapper to monitor and log RAM usage during the function execution.
    NOTE: can be applied to function but not to method of a class.
    Parameters:
        func: function
    """

    @wraps(func)
    def ram_usage_func(*args, **kwargs):
        ram, value = memory_usage(
            (func, args, *kwargs), interval=1.0, retval=True, max_usage=True
        )
        logging.info(
            f"Finished {func.__name__,}. Max RAM used {(ram / 1000):.4f} GB."
        )  # noqa
        return value

    return ram_usage_func
