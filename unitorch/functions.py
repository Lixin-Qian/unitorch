# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


def pop_first_non_none_value(
    *args,
    msg: Optional[str] = "default error msg",
):
    """
    Args:
        args: a list of python values
    Returns:
        return the first non-none value
    """
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError(f"{msg} can't find non-none value")
