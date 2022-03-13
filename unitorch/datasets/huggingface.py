# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import datasets
from itertools import cycle
from torch.utils import data
from datasets import load_dataset
from datasets import Dataset
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union


class hf_datasets(data.Dataset):
    """
    A dataclass of huggingface datasets library
    https://github.com/huggingface/datasets
    """

    def __init__(self, dataset: Dataset):
        """A class based on huggingface datasets
        `dataset` is an instance of huggingface dataset
        """
        self.__dataset__ = dataset

    @classmethod
    def from_csv(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        names: Optional[List[str]] = None,
        sep: Optional[str] = "\t",
        split: Optional[str] = None,
    ):
        """
        A classmethod of load csv/tsv/text files dataset
        Args:
            data_dir: defining the data_dir of the dataset configuration.
            data_files: path(s) to source data file(s).
            names: header names to data file(s).
            sep: seperator for text file(s).
            split: which split of the data to load.
        Returns: return a dataset.
        """
        if data_files is None:
            return

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return

        __dataset__ = load_dataset(
            "csv",
            data_dir=data_dir,
            data_files=data_files,
            delimiter=sep,
            column_names=names,
            quoting=3,
        )
        if split not in __dataset__:
            __dataset__ = __dataset__.get("train")
        else:
            __dataset__ = __dataset__.get(split)
        return cls(dataset=__dataset__)

    @classmethod
    def from_json(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        field: Optional[str] = None,
        split: Optional[str] = None,
    ):
        """
        A classmethod of load json files dataset
        Args:
            data_dir: defining the data_dir of the dataset configuration.
            data_files: path(s) to source data file(s).
            field: specify the field to load in json file.
            split: which split of the data to load.
        Returns: return a dataset.
        """
        if data_files is None:
            return

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return

        __dataset__ = load_dataset(
            "json",
            data_dir=data_dir,
            data_files=data_files,
            field=field,
        )
        if split not in __dataset__:
            __dataset__ = __dataset__.get("train")
        else:
            __dataset__ = __dataset__.get(split)
        return cls(dataset=__dataset__)

    @classmethod
    def from_parquet(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
    ):
        """
        A classmethod of load parquet files dataset
        Args:
            data_dir: defining the data_dir of the dataset configuration.
            data_files: path(s) to source data file(s).
            split: which split of the data to load.
        Returns: return a dataset.
        """
        if data_files is None:
            return

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return

        __dataset__ = load_dataset(
            "parquet",
            data_dir=data_dir,
            data_files=data_files,
        )
        if split not in __dataset__:
            __dataset__ = __dataset__.get("train")
        else:
            __dataset__ = __dataset__.get(split)
        return cls(dataset=__dataset__)

    @classmethod
    def from_hub(
        cls,
        data_name,
        config_name=None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
    ):
        """
        A classmethod of load hf hub dataset
        Args:
            data_name: a dataset repository on the hf hub.
            config_name: defining the name of the dataset configuration.
            data_dir: defining the data_dir of the dataset configuration.
            data_files: path(s) to source data file(s).
            split: which split of the data to load.
        Returns: return a dataset.
        """
        __dataset__ = load_dataset(
            data_name,
            name=config_name,
            data_dir=data_dir,
            data_files=data_files,
        )
        if split in __dataset__:
            __dataset__ = __dataset__.get(split)
            return cls(dataset=__dataset__)

    @property
    def dataset(self):
        """
        The property of actual hf dataset
        """
        return self.__dataset__

    def __getitem__(self, idx):
        return self.__dataset__[idx]

    def __len__(self):
        return len(self.__dataset__)


class hf_iterable_datasets(data.IterableDataset):
    """
    A dataclass of huggingface datasets library
    https://github.com/huggingface/datasets
    """

    def __init__(
        self,
        dataset: Dataset,
    ):
        self.__dataset__ = dataset

    def set_epoch(self, epoch):
        self.__dataset__.set_epoch(epoch)

    @classmethod
    def from_csv(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        names: Optional[List[str]] = None,
        sep: Optional[str] = "\t",
        split: Optional[str] = None,
    ):
        """
        A classmethod of load csv/tsv/text files dataset
        Args:
            data_dir: defining the data_dir of the dataset configuration.
            data_files: path(s) to source data file(s).
            names: header names to data file(s).
            sep: seperator for text file(s).
            split: which split of the data to load.
        Returns: return a streaming dataset.
        """
        if data_files is None:
            return

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return

        __dataset__ = load_dataset(
            "csv",
            data_dir=data_dir,
            data_files=data_files,
            delimiter=sep,
            column_names=names,
            quoting=3,
            streaming=True,
        )
        if split not in __dataset__:
            __dataset__ = __dataset__.get("train")
        else:
            __dataset__ = __dataset__.get(split)
        return cls(dataset=__dataset__)

    @classmethod
    def from_json(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        field: Optional[str] = None,
        split: Optional[str] = None,
    ):
        """
        A classmethod of load json files dataset
        Args:
            data_dir: defining the data_dir of the dataset configuration.
            data_files: path(s) to source data file(s).
            field: specify the field to load in json file.
            split: which split of the data to load.
        Returns: return a streaming dataset.
        """
        if data_files is None:
            return

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return

        __dataset__ = load_dataset(
            "json",
            data_dir=data_dir,
            data_files=data_files,
            field=field,
            streaming=True,
        )
        if split not in __dataset__:
            __dataset__ = __dataset__.get("train")
        else:
            __dataset__ = __dataset__.get(split)
        return cls(dataset=__dataset__)

    @classmethod
    def from_parquet(
        cls,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
    ):
        """
        A classmethod of load parquet files dataset
        Args:
            data_dir: defining the data_dir of the dataset configuration.
            data_files: path(s) to source data file(s).
            split: which split of the data to load.
        Returns: return a streaming dataset.
        """
        if data_files is None:
            return

        if isinstance(data_files, str) and not os.path.exists(data_files):
            return

        __dataset__ = load_dataset(
            "parquet",
            data_dir=data_dir,
            data_files=data_files,
            streaming=True,
        )
        if split not in __dataset__:
            __dataset__ = __dataset__.get("train")
        else:
            __dataset__ = __dataset__.get(split)
        return cls(dataset=__dataset__)

    @classmethod
    def from_hub(
        cls,
        data_name,
        config_name: Optional[str] = None,
        data_dir: Optional[str] = None,
        data_files: Optional[Union[str, List[str]]] = None,
        split: Optional[str] = None,
    ):
        """
        A classmethod of load hf hub dataset
        Args:
            data_name: a dataset repository on the hf hub.
            config_name: defining the name of the dataset configuration.
            data_dir: defining the data_dir of the dataset configuration.
            data_files: path(s) to source data file(s).
            split: which split of the data to load.
        Returns: return a streaming dataset
        """
        __dataset__ = load_dataset(
            data_name,
            name=config_name,
            data_dir=data_dir,
            data_files=data_files,
            streaming=True,
        )
        if split in __dataset__:
            __dataset__ = __dataset__.get(split)
            return cls(dataset=__dataset__)

    @property
    def dataset(self):
        """
        The property of actual hf dataset
        """
        return self.__dataset__

    def __iter__(self):
        for row_data in cycle(self.dataset):
            yield row_data
