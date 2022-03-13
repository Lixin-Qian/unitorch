# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import importlib
import pkg_resources


def load_template(template_path):
    module_name = f"unitorch.{template_path.replace('/', '.')}"
    importlib.import_module(module_name)
