# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import sys
import fire
import importlib
import unitorch.cli
from unitorch.cli import cached_path
from unitorch.cli import CoreConfigureParser, set_default_setting
from unitorch.cli import registered_task, init_registered_module
from unitorch_cli import load_template


@fire.decorators.SetParseFn(str)
def infer(config_path_or_template_dir: str, **kwargs):
    config_file = kwargs.pop("config_file", "config.ini")

    if os.path.isdir(config_path_or_template_dir):
        config_path = os.path.join(config_path_or_template_dir, config_file)
        sys.path.insert(0, config_path_or_template_dir)
        for f in os.listdir(config_path_or_template_dir):
            fpath = os.path.normpath(os.path.join(config_path_or_template_dir, f))
            if not f.startswith("_") and not f.startswith(".") and (f.endswith(".py") or os.path.isdir(fpath)):
                fname = f[:-3] if f.endswith(".py") else f
                module = importlib.import_module(f"{fname}")

    elif not config_path_or_template_dir.endswith(".ini"):
        load_template(config_path_or_template_dir)
        config_path = os.path.join(config_path_or_template_dir, config_file)
        config_path = cached_path(config_path)
        if config_path is None:
            config_path = cached_path(config_file)
    else:
        config_path = cached_path(config_path_or_template_dir)

    params = []
    for k, v in kwargs.items():
        if k.count("@") > 0:
            k0 = k.split("@")[0]
            k1 = "@".join(k.split("@")[1:])
        else:
            k0 = "core/cli"
            k1 = k
        params.append((k0, k1, v))

    if config_path is not None:
        config = CoreConfigureParser(config_path, params=params)
    else:
        config = CoreConfigureParser(params=params)

    set_default_setting(config)

    task_name = config.getdefault("core/cli", "task_name", None)
    depends_templates = config.getdefault("core/cli", "depends_templates", None)

    if depends_templates:
        for template in depends_templates:
            load_template(template)

    assert task_name is not None and task_name in registered_task
    cli_task = init_registered_module(task_name, config, registered_task)

    cli_task.infer()

    os._exit(0)


def cli_main():
    fire.Fire(infer)
