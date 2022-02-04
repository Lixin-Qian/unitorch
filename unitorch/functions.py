# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.


def pop_first_non_none_value(*args, msg="default error msg"):
    for arg in args:
        if arg is not None:
            return arg
    raise ValueError(f"{msg} can't find non-none value")
