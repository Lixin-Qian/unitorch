# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import importlib
import pkg_resources
from transformers.file_utils import is_remote_url
from unitorch import hf_cached_path


def get_config_file(config_file):
    if os.path.exists(config_file):
        return config_file

    if is_remote_url(config_file):
        return hf_cached_path(config_file)

    _config_file = pkg_resources.resource_filename("unitorch", config_file)
    if os.path.exists(_config_file):
        return _config_file


def load_template(template_path):
    module_name = f"unitorch.{template_path.replace('/', '.')}"
    importlib.import_module(module_name)
