# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import logging
import traceback
import importlib
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli.core import CoreClass, CoreConfigureParser


def rpartial(
    func,
    *args,
    **kwargs,
):
    return lambda *a, **kw: func(*(args + a), **dict(kwargs, **kw))


# default core config object
default_setting = CoreConfigureParser()


def get_default_setting():
    return default_setting


def set_default_setting(setting: Union[CoreConfigureParser, str]):
    if isinstance(setting, CoreConfigureParser):
        default_setting = setting
    elif os.path.exists(setting):
        default_setting = CoreConfigureParser(setting)
    else:
        raise ValueError(f"Can't set default settings by {setting}")


from unitorch.cli.decorators import (
    add_default_section_for_init,
    add_default_section_for_function,
)


# registry function
def registry_func(
    name: str,
    decorators: Union[Callable, List[Callable]] = None,
    save_dict: Dict = dict(),
):
    def actual_func(obj):
        save_dict[name] = dict(
            {
                "obj": obj,
                "decorators": decorators,
            }
        )
        return obj

    return actual_func


# register score/dataset/loss/model/optim/writer/scheduler/task
core_modules = [
    "score",
    "dataset",
    "loss",
    "model",
    "optim",
    "writer",
    "scheduler",
    "task",
]

for module in core_modules:
    globals()[f"registered_{module}"] = dict()
    globals()[f"register_{module}"] = partial(
        registry_func,
        save_dict=globals()[f"registered_{module}"],
    )

# register process function
registered_process = dict()


def get_import_module(import_file):
    modules = sys.modules.copy()
    for k, v in modules.items():
        if hasattr(v, "__file__") and v.__file__ == import_file:
            return v
    raise "can't find the module"


def register_process(
    name: str,
    decorators: Union[Callable, List[Callable]] = None,
):
    def actual_func(obj):
        trace_stacks = traceback.extract_stack()
        import_file = trace_stacks[-2][0]
        import_cls_name = trace_stacks[-2][2]
        import_module = get_import_module(import_file)
        registered_process[name] = dict(
            {
                "cls": {
                    "module": import_module,
                    "name": import_cls_name,
                },
                "obj": obj,
                "decorators": decorators,
            }
        )
        return obj

    return actual_func


# init registered modules
def init_registered_module(
    name: str,
    config: CoreConfigureParser,
    registered_module: Dict,
    **kwargs,
):
    if name not in registered_module:
        return

    v = registered_module[name]

    if v["decorators"]:
        return v["decorators"](v["obj"]).from_core_configure(config, **kwargs)
    return v["obj"].from_core_configure(config, **kwargs)


def init_registered_process(
    name: str,
    config: CoreConfigureParser,
    **kwargs,
):
    if name not in registered_process:
        return

    v = registered_process[name]
    cls = getattr(v["cls"]["module"], v["cls"]["name"])
    inst = cls.from_core_configure(config, **kwargs)
    if v["decorators"]:
        return rpartial(v["decorators"](v["obj"]), inst)
    else:
        return rpartial(v["obj"], inst)


# import cli modules
import unitorch.cli.datasets
import unitorch.cli.loss
import unitorch.cli.models
import unitorch.cli.optim
import unitorch.cli.scheduler
import unitorch.cli.score
import unitorch.cli.tasks
import unitorch.cli.writer
