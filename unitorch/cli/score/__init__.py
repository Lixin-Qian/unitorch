# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import torch.nn as nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from unitorch.score import (
    accuracy_score,
    recall_score,
    f1_score,
    bleu_score,
    voc_map_score,
    rouge1_score,
    rouge2_score,
    rougel_score,
    roc_auc_score,
    matthews_corrcoef,
    pearsonr,
    spearmanr,
)
from unitorch.cli import add_default_section_for_init, register_score
from unitorch.cli.models import (
    BaseOutputs,
    BaseTargets,
    ClassificationOutputs,
    ClassificationTargets,
    GenerationOutputs,
    GenerationTargets,
    DetectionOutputs,
    DetectionTargets,
    SegmentationOutputs,
    SegmentationTargets,
    LossOutputs,
)


class Score(nn.Module):
    pass


@register_score("core/score/acc")
class AccuracyScore(Score):
    def __init__(self, gate: float = 0.5):
        super().__init__()
        self.gate = gate

    @classmethod
    @add_default_section_for_init("core/score/acc")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[ClassificationOutputs, GenerationOutputs, SegmentationOutputs],
        targets: Union[ClassificationTargets, GenerationTargets, SegmentationTargets],
    ):
        if hasattr(outputs, "outputs"):
            outputs = outputs.outputs

        if hasattr(targets, "targets"):
            targets = targets.targets

        if isinstance(outputs, GenerationOutputs):
            outputs = outputs.sequences
            outputs = outputs.view(-1, output.size(-1))

        if isinstance(targets, GenerationTargets):
            targets = targets.refs
            targets = targets.view(-1)

        if isinstance(outputs, SegmentationOutputs):
            outputs = outputs.masks
            outputs = torch.cat([t.view(-1) for t in outputs])

        if isinstance(targets, SegmentationTargets):
            targets = targets.targets
            targets = torch.cat([t.view(-1) for t in targets])

        if outputs.dim() == 2:
            outputs = outputs.argmax(dim=-1) if outputs.size(-1) > 1 else outputs[:, 0] > self.gate

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return accuracy_score(targets, outputs)


@register_score("core/score/rec")
class RecallScore(Score):
    def __init__(self, gate: float = 0.5):
        super().__init__()
        self.gate = gate

    @classmethod
    @add_default_section_for_init("core/score/rec")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, ClassificationOutputs],
        targets: Union[BaseTargets, ClassificationTargets],
    ):
        if hasattr(outputs, "outputs"):
            outputs = outputs.outputs

        if hasattr(targets, "targets"):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = outputs.argmax(dim=-1) if outputs.size(-1) > 1 else outputs[:, 0] > self.gate

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return recall_score(targets, outputs)


@register_score("core/score/f1")
class F1Score(Score):
    def __init__(self, gate: float = 0.5):
        super().__init__()
        self.gate = gate

    @classmethod
    @add_default_section_for_init("core/score/f1")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, ClassificationOutputs],
        targets: Union[BaseTargets, ClassificationTargets],
    ):
        if hasattr(outputs, "outputs"):
            outputs = outputs.outputs

        if hasattr(targets, "targets"):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = outputs.argmax(dim=-1) if outputs.size(-1) > 1 else outputs[:, 0] > self.gate

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return f1_score(targets, outputs)


@register_score("core/score/auc")
class AUCScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/auc")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, ClassificationOutputs],
        targets: Union[BaseTargets, ClassificationTargets],
    ):
        if hasattr(outputs, "outputs"):
            outputs = outputs.outputs

        if hasattr(targets, "targets"):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = outputs[:, 1] if outputs.size(-1) > 1 else outputs[:, 0]

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return roc_auc_score(targets, outputs)


@register_score("core/score/mattcorr")
class MattCorrScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/mattcorr")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, ClassificationOutputs],
        targets: Union[BaseTargets, ClassificationTargets],
    ):
        if hasattr(outputs, "outputs"):
            outputs = outputs.outputs

        if hasattr(targets, "targets"):
            targets = targets.targets

        if outputs.dim() == 2:
            outputs = outputs.argmax(dim=-1) if outputs.size(-1) > 1 else outputs[:, 0] > self.gate

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return matthews_corrcoef(targets, outputs)


@register_score("core/score/pearsonr_corr")
class PearsonrCorrScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/pearsonr_corr")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, ClassificationOutputs],
        targets: Union[BaseTargets, ClassificationTargets],
    ):
        if hasattr(outputs, "outputs"):
            outputs = outputs.outputs

        if hasattr(targets, "targets"):
            targets = targets.targets

        if outputs.dim() == 2 and outputs.size(-1) == 1:
            outputs = outputs[:, 0]

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return pearsonr(targets, outputs)[0]


@register_score("core/score/spearmanr_corr")
class SpearmanrCorrScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/spearmanr_corr")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, ClassificationOutputs],
        targets: Union[BaseTargets, ClassificationTargets],
    ):
        if hasattr(outputs, "outputs"):
            outputs = outputs.outputs

        if hasattr(targets, "targets"):
            targets = targets.targets

        if outputs.dim() == 2 and outputs.size(-1) == 1:
            outputs = outputs[:, 0]

        if targets.dim() == 2 and targets.size(-1) == 1:
            targets = targets[:, 0]

        assert outputs.dim() == 1 and targets.dim() == 1

        return spearmanr(targets, outputs)[0]


@register_score("core/score/bleu")
class BleuScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/bleu")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, GenerationOutputs],
        targets: Union[BaseTargets, GenerationTargets],
    ):
        if hasattr(outputs, "sequences"):
            outputs = outputs.sequences

        if hasattr(targets, "refs"):
            targets = targets.refs

        return bleu_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=[0, 1],
        )


@register_score("core/score/rouge1")
class Rouge1Score(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/rouge1")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, GenerationOutputs],
        targets: Union[BaseTargets, GenerationTargets],
    ):
        if hasattr(outputs, "sequences"):
            outputs = outputs.sequences

        if hasattr(targets, "refs"):
            targets = targets.refs

        return rouge1_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=[0, 1],
        )["f1"]


@register_score("core/score/rouge2")
class Rouge2Score(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/rouge2")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, GenerationOutputs],
        targets: Union[BaseTargets, GenerationTargets],
    ):
        if hasattr(outputs, "sequences"):
            outputs = outputs.sequences

        if hasattr(targets, "refs"):
            targets = targets.refs

        return rouge2_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=[0, 1],
        )["f1"]


@register_score("core/score/rougel")
class RougelScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/rougel")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, GenerationOutputs],
        targets: Union[BaseTargets, GenerationTargets],
    ):
        if hasattr(outputs, "sequences"):
            outputs = outputs.sequences

        if hasattr(targets, "refs"):
            targets = targets.refs

        return rougel_score(
            targets.long(),
            outputs.long(),
            ignore_tokens=[0, 1],
        )["f1"]


@register_score("core/score/loss")
class LossScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/rougel")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[BaseOutputs, LossOutputs],
        targets: Union[BaseTargets, GenerationTargets],
    ):
        if hasattr(outputs, "loss"):
            loss = outputs.loss

        return -float(torch.mean(loss))


@register_score("core/score/voc_map")
class VOCMAPScore(Score):
    def __init__(
        self,
    ):
        super().__init__()

    @classmethod
    @add_default_section_for_init("core/score/voc_map")
    def from_core_configure(cls, config, **kwargs):
        pass

    def forward(
        self,
        outputs: Union[DetectionOutputs],
        targets: Union[DetectionTargets],
    ):
        p_bboxes = outputs.bboxes
        p_scores = outputs.scores
        p_classes = outputs.classes
        gt_bboxes = targets.bboxes
        gt_classes = targets.classes
        return voc_map_score(
            p_bboxes=[t.numpy() for t in p_bboxes],
            p_scores=[t.numpy() for t in p_scores],
            p_classes=[t.numpy() for t in p_classes],
            gt_bboxes=[t.numpy() for t in gt_bboxes],
            gt_classes=[t.numpy() for t in gt_classes],
        )
