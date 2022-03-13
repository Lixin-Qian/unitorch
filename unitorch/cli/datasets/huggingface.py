# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import ast
import torch
import torch.distributed as dist
from copy import deepcopy
from datasets import Dataset
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.cli import cached_path
from unitorch.datasets.huggingface import hf_datasets, hf_iterable_datasets
from unitorch.cli import CoreClass, CoreConfigureParser
from unitorch.cli import registered_process, register_dataset
from unitorch.cli import add_default_section_for_init, add_default_section_for_function
from unitorch.cli import init_registered_process
from unitorch.cli.models import BaseInputs, BaseTargets


class ast_function(object):
    def __init__(self, func: str):
        for name in registered_process.keys():
            func = func.replace(name, name.replace("/", "_"))

        registered_process_mapping = {k.replace("/", "_"): k for k, v in registered_process.items()}
        self.func = func
        self.__ast_func__ = ast.parse(func, "", mode="eval")
        self.__ast_keys__ = []
        self.__ast_process__ = []
        for node in ast.walk(self.__ast_func__):
            if isinstance(node, ast.Name):
                if node.id in registered_process_mapping:
                    self.__ast_process__.append(node.id)
                else:
                    self.__ast_keys__.append(node.id)
        self.__ast_func__ = compile(self.__ast_func__, "", "eval")

    def process(self, row: Dict):
        for key in self.__ast_keys__:
            if key not in row:
                continue
            locals()[key] = row.get(key, None)

        return eval(self.__ast_func__)


class ast_dataset(hf_datasets):
    def __init__(
        self,
        dataset: Dataset,
        process_functions: List[ast_function],
    ):
        super().__init__(dataset=dataset)
        self.process_functions = deepcopy(process_functions)

    def __getitem__(self, idx):
        multi_inputs, multi_targets = [], []
        row = self.dataset[idx]
        for pfunc in self.process_functions:
            results = pfunc.process(row)

            if isinstance(results, BaseInputs) or isinstance(results, BaseTargets):
                results = [results]

            for result in results:
                if isinstance(result, BaseInputs):
                    is_new = True
                    for _inputs in multi_inputs:
                        if type(_inputs) == type(result):
                            _inputs.update(result)
                            is_new = False
                            break

                    if is_new:
                        multi_inputs.append(result)

                if isinstance(result, BaseTargets):
                    is_new = True
                    for _targets in multi_targets:
                        if type(_targets) == type(result):
                            _targets.update(result)
                            is_new = False
                            break

                    if is_new:
                        multi_targets.append(result)

        if len(multi_inputs) == 0:
            multi_inputs = BaseInputs()
        elif len(multi_inputs) == 1:
            multi_inputs = multi_inputs[0]

        if len(multi_targets) == 0:
            multi_targets = BaseTargets()
        elif len(multi_targets) == 1:
            multi_targets = multi_targets[0]

        return multi_inputs, multi_targets

    def __len__(
        self,
    ):
        return len(self.dataset)


class ast_iterable_dataset(hf_iterable_datasets):
    def __init__(
        self,
        dataset: Dataset,
        process_functions: List[ast_function],
        enable_ddp_partition: bool = True,
    ):
        super().__init__(dataset=dataset)
        self.process_functions = deepcopy(process_functions)
        self.dataset.shuffle(10000)
        if enable_ddp_partition and dist.is_initialized():
            self.global_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.global_rank = 0
            self.world_size = 1

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        mod = self.world_size
        shift = self.global_rank
        if worker_info:
            mod *= worker_info.num_workers
            shift = self.global_rank * worker_info.num_workers + worker_info.id
        for i, row in enumerate(self.dataset):
            if (i + shift) % mod != 0:
                continue
            inputs, targets = None, None
            for pfunc in self.process_functions:
                results = pfunc.process(row)

                if isinstance(results, BaseInputs) or isinstance(results, BaseTargets):
                    results = [results]

                for result in results:
                    if isinstance(result, BaseInputs):
                        if inputs is None:
                            inputs = result
                        else:
                            inputs.update(result)

                    if isinstance(result, BaseTargets):
                        if targets is None:
                            targets = result
                        else:
                            targets.update(result)
            if inputs is None:
                inputs = BaseInputs()

            if targets is None:
                targets = BaseTargets()

            yield inputs, targets

    def set_skip_step(self, step):
        self.__dataset__ = self.__dataset__.skip(step * self.world_size)


@register_dataset("core/dataset/ast")
class ast_datasets(CoreClass):
    """
    ast datasets for cli mode, using ast to parse str to preprocess functions
    """

    splits = ["train", "dev", "test"]
    templates = ["csv", "json", "parquet", "hub"]
    __ast_datasets__ = dict()

    def __init__(self, __config__: CoreConfigureParser):
        self.config = __config__

    def __getdataset__(self, split):
        config = self.config

        registered_process_mapping = {k.replace("/", "_"): k for k, v in registered_process.items()}

        config.set_default_section(f"core/dataset/ast")
        _iterable = config.getoption("iterable", False)
        _template = config.getoption("template", "csv")
        _data_name = config.getoption("data_name", None)
        _config_name = config.getoption("config_name", None)
        _data_dir = config.getoption("data_dir", None)
        _data_files = config.getoption("data_files", None)
        _names = config.getoption("names", None)
        _sep = config.getoption("sep", "\t")
        _field = config.getoption("field", None)
        _process_functions = config.getoption("preprocess_functions", None)
        _enable_ddp_partition = config.getoption("enable_ddp_partition", True)

        _hf_datasets = hf_iterable_datasets if _iterable else hf_datasets
        _ast_dataset = ast_iterable_dataset if _iterable else ast_dataset

        config.set_default_section(f"core/dataset/ast/{split}")

        template = config.getoption("template", _template)
        if config.getoption("data_name", _data_name) is not None:
            template = "hub"

        assert template in self.templates

        new_split = "validation" if split == "dev" else split
        new_split = config.getoption("split", new_split)

        # get dataset
        dataset = None
        if template == "csv":
            data_dir = config.getoption("data_dir", _data_dir)
            data_files = config.getoption("data_files", _data_files)
            names = config.getoption("names", _names)
            sep = config.getoption("sep", _sep)
            dataset = _hf_datasets.from_csv(
                data_dir=data_dir,
                data_files=data_files,
                names=names,
                sep=sep,
                split=new_split,
            )

        if template == "json":
            data_dir = config.getoption("data_dir", _data_dir)
            data_files = config.getoption("data_files", _data_files)
            field = config.getoption("field", _field)

            dataset = _hf_datasets.from_json(
                data_dir=data_dir,
                data_files=data_files,
                field=field,
                split=new_split,
            )

        if template == "parquet":
            data_dir = config.getoption("data_dir", _data_dir)
            data_files = config.getoption("data_files", _data_files)
            dataset = _hf_datasets.from_parquet(
                data_dir=data_dir,
                data_files=data_files,
                split=new_split,
            )

        if template == "hub":
            data_name = config.getoption("data_name", _data_name)
            config_name = config.getoption("config_name", _config_name)
            data_dir = config.getoption("data_dir", _data_dir)
            data_files = config.getoption("data_files", _data_files)
            data_name = cached_path(data_name) if data_name.endswith(".py") else data_name
            dataset = _hf_datasets.from_hub(
                data_name=data_name,
                config_name=config_name,
                data_dir=data_dir,
                data_files=data_files,
                split=new_split,
            )

        assert dataset is not None

        # get process functions
        process_functions = config.getoption("preprocess_functions", _process_functions)
        if process_functions is None:
            process_functions = []
        else:
            process_functions = [ast_function(func) for func in process_functions]

        for pfunc in process_functions:
            for name in pfunc.__ast_process__:
                globals()[name] = init_registered_process(
                    registered_process_mapping[name],
                    config,
                )

        enable_ddp_partition = config.getoption("enable_ddp_partition", _enable_ddp_partition)

        if isinstance(_ast_dataset, hf_iterable_datasets):
            self.__ast_datasets__[split] = _ast_dataset(
                dataset=dataset.dataset,
                process_functions=process_functions,
                enable_ddp_partition=enable_ddp_partition,
            )
        else:
            self.__ast_datasets__[split] = _ast_dataset(
                dataset=dataset.dataset,
                process_functions=process_functions,
            )

        return self.__ast_datasets__.get(split)

    @classmethod
    @add_default_section_for_init("core/dataset/ast")
    def from_core_configure(cls, config, **kwargs):
        return cls(__config__=config)

    def get(self, split: str = "train"):
        return self.__getdataset__(split)
