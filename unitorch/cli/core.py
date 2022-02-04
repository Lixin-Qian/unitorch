# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import os
import logging
import ast
from abc import ABCMeta
from abc import abstractmethod
import torch
import torch.nn as nn
from transformers.file_utils import is_remote_url
from unitorch import hf_cached_path
import configparser

# core class
class CoreClass(metaclass=ABCMeta):
    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def from_core_configure(cls, config, **kwargs):
        pass


# core config class
class CoreConfigureParser(configparser.ConfigParser):
    def __init__(self, fpath="./config.ini", params=[]):
        super().__init__(interpolation=configparser.ExtendedInterpolation())
        self.fpath = fpath
        self.read(fpath)
        for param in params:
            assert len(param) == 3
            k0, k1, v = param
            if not self.has_section(k0):
                self.add_section(k0)
            self.set(k0, k1, str(v))

        self._freeze_section = None
        self._map_dict = dict()

        self._default_section = self.getdefault("core/config", "default_section", None)
        self._freeze_section = self.getdefault("core/config", "freeze_section", None)
        self._map_dict = self.getdefault("core/config", "section_map_dict", None)
        if self._map_dict is not None:
            self._map_dict = eval(self._map_dict)
        else:
            self._map_dict = dict()

    def _getdefault(self, section, option, value=None):
        if not self.has_section(section):
            return value
        if self.has_option(section, option):
            return self.get(section, option)
        return value

    def _ast_replacement(self, node):
        value = node.id
        if value in ("True", "False", "None"):
            return node
        return ast.Str(value)

    def _ast_literal_eval(self, value):
        root = ast.parse(value, mode="eval")
        if isinstance(root.body, ast.BinOp):
            raise ValueError(value)

        for node in ast.walk(root):
            for field, child in ast.iter_fields(node):
                if isinstance(child, list):
                    for index, subchild in enumerate(child):
                        if isinstance(subchild, ast.Name):
                            child[index] = self._ast_replacement(subchild)
                elif isinstance(child, ast.Name):
                    replacement = self._ast_replacement(child)
                    node.__setattr__(field, replacement)
        return ast.literal_eval(root)

    def get(
        self,
        section,
        option,
        raw=False,
        vars=None,
        fallback=configparser._UNSET,
    ):
        value = super().get(
            section,
            option,
            raw=raw,
            vars=vars,
            fallback=fallback,
        )
        try:
            return self._ast_literal_eval(value)
        except (SyntaxError, ValueError):
            return value

    def set_map_dict(self, map_dict=dict()):
        self._map_dict = map_dict

    def set_freeze_section(self, section):
        self._freeze_section = section

    def set_default_section(self, section):
        self._default_section = section

    def getdefault(self, section, option, value=None):
        value = self._getdefault(section, option, value)

        if self._freeze_section:
            value = self._getdefault(self._freeze_section, option, value)

        if self._map_dict.get(section):
            section = self._map_dict.get(section)
            value = self._getdefault(section, option, value)
        return value

    def getoption(self, option, value=None):
        return self.getdefault(self._default_section, option, value)

    def print(self):
        print("#" * 30, "Config Info".center(20, " "), "#" * 30)
        for sec, item in self.items():
            for k, v in item.items():
                print(
                    sec.rjust(10, " "),
                    ":".center(5, " "),
                    k.ljust(30, " "),
                    ":".center(5, " "),
                    v.ljust(30, " "),
                )
        print("#" * 30, "Config Info End".center(20, " "), "#" * 30)

    def save(self, save_path="./config.ini"):
        self.write(open(save_path, "w"))
        return save_path

    def copy(self, config):
        setting = [
            (sec, k, v) for sec in config.sections() for k, v in config[sec].items()
        ]
        return type(self)(self.fpath, setting)
