# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.writer import (
    GeneralWriter as _GeneralWriter,
    GeneralCSVWriter as _GeneralCSVWriter,
)
from unitorch.cli import add_default_section_for_init, register_writer
from unitorch.cli.models import (
    BaseOutputs,
    ClassificationOutputs,
    GenerationOutputs,
)


@register_writer("core/writer/default")
class GeneralWriter(_GeneralWriter):
    def _init___(self):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/writer/default")
    def from_core_configure(cls, config, **kwargs):
        pass

    def __call__(self, outputs: BaseOutputs):
        return super().__call__(outputs.to_dict())


@register_writer("core/writer/csv")
class GeneralCSVWriter(_GeneralCSVWriter):
    def __init__(
        self,
        headers: Optional[List[str]] = None,
        sep: str = "\t",
    ):
        super().__init__(
            headers=headers,
            sep=sep,
        )

    @classmethod
    @add_default_section_for_init("core/writer/csv")
    def from_core_configure(cls, config, **kwargs):
        pass

    def __call__(self, outputs: BaseOutputs):
        return super().__call__(outputs.to_dict())


# more writers
import unitorch.cli.writer.generation
