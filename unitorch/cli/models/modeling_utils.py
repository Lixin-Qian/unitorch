# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


class BaseObjects(object):
    def __init__(
        self,
        attrs: Dict = dict(),
        **kwargs,
    ):
        for k, v in type(self).__dict__.items():
            if not callable(getattr(self, k)) and not k.startswith("__"):
                attrs[k] = attrs.get(k, v)
        self.set_attrs({**attrs, **kwargs})

    def set_attrs(self, attrs: Dict = dict()):
        for k, v in attrs.items():
            setattr(self, k, v)

    @classmethod
    def from_dict(cls, attrs: Dict = dict()):
        return cls(attrs)

    @classmethod
    def from_list(cls, *attrs, dim=0, op="stack"):
        def _func(*args, dim=0, op="stack"):
            if args[0] is None:
                return None
            if isinstance(args[0], dict):
                keys = args[0].keys()
                return dict(
                    {k: _func(*[arg[k] for arg in args], dim=dim, op=op) for k in keys}
                )
            if isinstance(args[0], list) or isinstance(args[0], tuple):
                return list([_func(*arg, dim=dim, op=op) for arg in zip(*args)])
            if isinstance(args[0], torch.Tensor) and op == "stack":
                assert all(args[i].shape == args[0].shape for i in range(1, len(args)))
                return torch.stack(args, dim=dim)
            if isinstance(args[0], torch.Tensor) and op == "concat":
                return torch.cat(args, dim=dim)

            raise ValueError(f"{args} can't stack or concat")

        return cls(
            _func(
                *[attr if isinstance(attr, dict) else attr.to_dict() for attr in attrs],
                dim=dim,
                op=op,
            )
        )

    def add_dict(self, attrs: Dict = dict()):
        for k, v in attrs.items():
            setattr(self, k, v)

    def to_dict(self):
        items = {
            k: v
            for k, v in self.__dict__.items()
            if not callable(getattr(self, k)) and not k.startswith("__")
        }
        return items

    def update(self, inputs):
        if isinstance(inputs, dict):
            self.add_dict(inputs)
        else:
            self.add_dict(inputs.to_dict())

    # for ddp evaluation
    def sync(self):
        def _func(attr):
            if attr is None:
                return None

            if isinstance(attr, torch.Tensor):
                new_attr = [attr.clone() for _ in range(dist.get_world_size())]
                dist.all_gather(new_attr, attr)
                return torch.cat(new_attr)

            if isinstance(attr, tuple) or isinstance(attr, list):
                return list(_func(b) for b in attr)

            keys = attr.keys()
            return dict({k: _func(attr[k]) for k in keys})

        attrs = _func(self.to_dict())
        return type(self)(attrs)

    def cuda(self):
        def _func(attr):
            if attr is None:
                return None

            if isinstance(attr, torch.Tensor):
                return attr.cuda(non_blocking=True)

            if isinstance(attr, tuple) or isinstance(attr, list):
                return list(_func(b) for b in attr)

            keys = attr.keys()
            return dict({k: _func(attr[k]) for k in keys})

        attrs = _func(self.to_dict())
        return type(self)(attrs)

    def cpu(self):
        def _func(attr):
            if attr is None:
                return None

            if isinstance(attr, torch.Tensor):
                return attr.cpu()

            if isinstance(attr, tuple) or isinstance(attr, list):
                return list(_func(b) for b in attr)

            keys = attr.keys()
            return dict({k: _func(attr[k]) for k in keys})

        attrs = _func(self.to_dict())
        return type(self)(attrs)


class BaseInputs(BaseObjects):
    def __init__(
        self,
        attrs: Dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)


class BaseOutputs(BaseObjects):
    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)


class BaseTargets(BaseObjects):
    sample_weight: torch.Tensor = None

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert self.sample_weight is None or isinstance(
            self.sample_weight, torch.Tensor
        )


class ListInputs(BaseInputs):
    def __init__(
        self,
        attrs: Dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)

    @classmethod
    def from_list(cls, *attrs, **kwargs):
        attrs = [attr.to_dict() for attr in attrs]
        first = attrs[0]
        keys = first.keys()
        new_attrs = dict()
        for k in keys:
            if first[k] is None:
                continue
            new_attrs[k] = [attr[k] for attr in attrs]
        return cls(new_attrs)


class LossOutputs(BaseOutputs):
    loss: torch.Tensor

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.loss, torch.Tensor)


class EmbeddingOutputs(BaseOutputs):
    embedding: torch.Tensor

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.embedding, torch.Tensor)


class ClassificationOutputs(BaseOutputs):
    outputs: torch.Tensor

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.outputs, torch.Tensor)


class ClassificationTargets(BaseTargets):
    targets: torch.Tensor

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.targets, torch.Tensor)


class GenerationOutputs(BaseOutputs):
    sequences: torch.Tensor
    sequences_scores: torch.Tensor = None

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.sequences, torch.Tensor)
        assert self.sequences_scores is None or isinstance(
            self.sequences_scores, torch.Tensor
        )


class GenerationTargets(BaseTargets):
    refs: torch.Tensor
    masks: torch.Tensor = None

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.refs, torch.Tensor)
        assert self.masks is None or isinstance(self.masks, torch.Tensor)


class DetectionOutputs(BaseOutputs):
    bboxes: Union[torch.Tensor, List[torch.Tensor]]
    scores: Union[torch.Tensor, List[torch.Tensor]]
    features: Union[torch.Tensor, List[torch.Tensor]]
    classes: Union[torch.Tensor, List[torch.Tensor]]

    __special_fields__ = ["bboxes", "scores", "features", "classes"]

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.bboxes, torch.Tensor) or (
            (isinstance(self.bboxes, list) or isinstance(self.bboxes, tuple))
            and isinstance(self.bboxes[0], torch.Tensor)
        )
        assert isinstance(self.scores, torch.Tensor) or (
            (isinstance(self.scores, list) or isinstance(self.scores, tuple))
            and isinstance(self.scores[0], torch.Tensor)
        )
        if not hasattr(self, "features"):
            self.features = None
        assert (
            self.features is None
            or isinstance(self.features, torch.Tensor)
            or (
                (isinstance(self.features, list) or isinstance(self.features, tuple))
                and isinstance(self.features[0], torch.Tensor)
            )
        )
        assert isinstance(self.classes, torch.Tensor) or (
            (isinstance(self.classes, list) or isinstance(self.classes, tuple))
            and isinstance(self.classes[0], torch.Tensor)
        )

    @classmethod
    def from_list(cls, *attrs, dim=0, op="stack"):
        attrs = [attr.to_dict() for attr in attrs]
        first = attrs[0]
        new_attrs = dict()
        for k in first.keys():
            if isinstance(first[k], torch.Tensor):
                v = [attr[k] for attr in attrs]
            elif isinstance(first[k], list) or isinstance(first[k], tuple):
                assert isinstance(first[k][0], torch.Tensor)
                v = list(chain.from_iterable([attr[k] for attr in attrs]))
            else:
                v = None
            new_attrs[k] = v
        return cls(new_attrs)

    def sync(self):
        attrs = dict()
        for k, v in self.to_dict().items():
            if isinstance(v, torch.Tensor):
                new_attr = [v.clone() for _ in range(dist.get_world_size())]
                dist.all_gather(new_attr, v)
            elif isinstance(v, list) or isinstance(v, tuple):
                assert isinstance(v[0], torch.Tensor)
                new_attr = [[] for _ in range(dist.get_world_size())]
                dist.all_gather_object(new_attr, v)
                new_attr = list(chain.from_iterable(new_attr))
            else:
                new_attr = None
            attrs[k] = new_attr
        return type(self)(attrs)


class DetectionTargets(BaseTargets):
    bboxes: Union[torch.Tensor, List[torch.Tensor]]
    classes: Union[torch.Tensor, List[torch.Tensor]]

    __special_fields__ = ["bboxes", "classes"]

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.bboxes, torch.Tensor) or (
            (isinstance(self.bboxes, list) or isinstance(self.bboxes, tuple))
            and isinstance(self.bboxes[0], torch.Tensor)
        )
        assert isinstance(self.classes, torch.Tensor) or (
            (isinstance(self.classes, list) or isinstance(self.classes, tuple))
            and isinstance(self.classes[0], torch.Tensor)
        )

    @classmethod
    def from_list(cls, *attrs, dim=0, op="stack"):
        attrs = [attr.to_dict() for attr in attrs]
        first = attrs[0]
        new_attrs = dict()
        for k in first.keys():
            if isinstance(first[k], torch.Tensor):
                v = [attr[k] for attr in attrs]
            elif isinstance(first[k], list) or isinstance(first[k], tuple):
                assert isinstance(first[k][0], torch.Tensor)
                v = list(chain.from_iterable([attr[k] for attr in attrs]))
            else:
                v = None
            new_attrs[k] = v
        return cls(new_attrs)

    def sync(self):
        attrs = dict()
        for k, v in self.to_dict().items():
            if isinstance(v, torch.Tensor):
                new_attr = [v.clone() for _ in range(dist.get_world_size())]
                dist.all_gather(new_attr, v)
            elif isinstance(v, list) or isinstance(v, tuple):
                assert isinstance(v[0], torch.Tensor)
                new_attr = [[] for _ in range(dist.get_world_size())]
                dist.all_gather_object(new_attr, v)
                new_attr = list(chain.from_iterable(new_attr))
            else:
                new_attr = None
            attrs[k] = new_attr
        return type(self)(attrs)


class SegmentationOutputs(DetectionOutputs):
    masks: Union[torch.Tensor, List[torch.Tensor]]

    __special_fields__ = ["masks"]

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.masks, torch.Tensor) or (
            (isinstance(self.masks, list) or isinstance(self.masks, tuple))
            and isinstance(self.masks[0], torch.Tensor)
        )


class SegmentationTargets(DetectionTargets):
    targets: Union[torch.Tensor, List[torch.Tensor]]

    __special_fields__ = ["targets"]

    def __init__(
        self,
        attrs: dict = dict(),
        **kwargs,
    ):
        super().__init__(attrs, **kwargs)
        assert isinstance(self.targets, torch.Tensor) or (
            (isinstance(self.targets, list) or isinstance(self.targets, tuple))
            and isinstance(self.targets[0], torch.Tensor)
        )


# decorators for models
def general_model_decorator(cls):
    class GeneralModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__generation_model__" in kwargs:
                self.model = kwargs.pop("__general_model__")
            else:
                self.model = cls(*args, **kwargs)

            __more_attrs__ = [
                "load_state_dict",
                "state_dict",
                "save_checkpoint",
                "from_checkpoint",
                "from_pretrained",
            ]
            for __more_attr__ in __more_attrs__:
                setattr(self, __more_attr__, getattr(self.model, __more_attr__))

        def forward(self, *args, **kwargs):
            if self.training:
                return self.model(*args, **kwargs)
            return self.model.inference(*args, **kwargs)

        @classmethod
        def from_core_configure(_cls, cfg, **kwargs):
            model = cls.from_core_configure(cfg, **kwargs)
            return _cls(__general_model__=model)

    return GeneralModel


def generation_model_decorator(cls):
    class GenerationModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__generation_model__" in kwargs:
                self.model = kwargs.pop("__generation_model__")
            else:
                self.model = cls(*args, **kwargs)

            __more_attrs__ = [
                "load_state_dict",
                "state_dict",
                "save_checkpoint",
                "from_checkpoint",
                "from_pretrained",
            ]
            for __more_attr__ in __more_attrs__:
                setattr(self, __more_attr__, getattr(self.model, __more_attr__))

        def forward(self, *args, **kwargs):
            if self.training:
                return self.model(*args, **kwargs)
            return self.model.generate(*args, **kwargs)

        @classmethod
        def from_core_configure(_cls, cfg, **kwargs):
            model = cls.from_core_configure(cfg, **kwargs)
            return _cls(__generation_model__=model)

    return GenerationModel


def detection_model_decorator(cls):
    class DetectionModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__detection_model__" in kwargs:
                self.model = kwargs.pop("__detection_model__")
            else:
                self.model = cls(*args, **kwargs)

            __more_attrs__ = [
                "load_state_dict",
                "state_dict",
                "save_checkpoint",
                "from_checkpoint",
                "from_pretrained",
            ]
            for __more_attr__ in __more_attrs__:
                setattr(self, __more_attr__, getattr(self.model, __more_attr__))

        def forward(self, *args, **kwargs):
            if self.training:
                return self.model(*args, **kwargs)
            return self.model.detect(*args, **kwargs)

        @classmethod
        def from_core_configure(_cls, cfg, **kwargs):
            model = cls.from_core_configure(cfg, **kwargs)
            return _cls(__detection_model__=model)

    return DetectionModel


def segmentation_model_decorator(cls):
    class SegmentationModel(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            if "__segmentation_model__" in kwargs:
                self.model = kwargs.pop("__segmentation_model__")
            else:
                self.model = cls(*args, **kwargs)

            __more_attrs__ = [
                "load_state_dict",
                "state_dict",
                "save_checkpoint",
                "from_checkpoint",
                "from_pretrained",
            ]
            for __more_attr__ in __more_attrs__:
                setattr(self, __more_attr__, getattr(self.model, __more_attr__))

        def forward(self, *args, **kwargs):
            if self.training:
                return self.model(*args, **kwargs)
            return self.model.segment(*args, **kwargs)

        @classmethod
        def from_core_configure(_cls, cfg, **kwargs):
            model = cls.from_core_configure(cfg, **kwargs)
            return _cls(__segmentation_model__=model)

    return SegmentationModel
