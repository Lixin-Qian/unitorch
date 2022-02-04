# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import unitorch
import torch
from absl.testing import absltest, parameterized
from unitorch.cli import CoreConfigureParser
from unitorch.cli import (
    add_default_section_for_init,
    add_default_section_for_function,
)


class SimpleClass(object):
    def __init__(self, param1=1, param2=2, param3=3):
        self.param1 = param1
        self.param2 = param2
        self.param3 = param3

    @classmethod
    @add_default_section_for_init("simple_class")
    def from_core_configure(cls, config, **kwargs):
        pass

    @add_default_section_for_function("simple_class")
    def inst_function(self, param4=4, param5="param5"):
        self.param4 = param4
        self.param5 = param5


class DecoratorsTest(parameterized.TestCase):
    def setUp(self):
        self.config = CoreConfigureParser(
            params=[
                ["simple_class", "param1", 2],
                ["simple_class", "param2", 3],
                ["simple_class", "param5", "param6"],
            ]
        )

    def test_default_for_init(self):
        inst = SimpleClass.from_core_configure(self.config)

        assert inst.param1 == 2 and inst.param2 == 3 and inst.param3 == 3

    def test_default_for_function(self):
        inst = SimpleClass.from_core_configure(self.config)
        inst.inst_function()

        assert inst.param4 == 4 and inst.param5 == "param6"
