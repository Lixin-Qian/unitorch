# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import importlib
from functools import partial
from unitorch.cli import CoreConfigureParser, registry_func


class service_module(object):
    def __init__(self, config: CoreConfigureParser):
        pass

    def start(self, **kwargs):
        pass

    def stop(self, **kwargs):
        pass

    def restart(self, **kwargs):
        pass


registered_service = dict()
register_service = partial(
    registry_func,
    save_dict=registered_service,
)

# import service modules
