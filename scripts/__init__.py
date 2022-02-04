# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import importlib
from functools import partial
from unitorch.cli import CoreConfigureParser, registry_func


class script_module(object):
    def __init__(self, config: CoreConfigureParser):
        pass

    def run(self, **kwargs):
        pass


registered_script = dict()
register_script = partial(
    registry_func,
    save_dict=registered_script,
)

# import script modules
