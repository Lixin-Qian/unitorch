import numpy as np
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union


def _voc_ap(
    rec,
    prec,
    use_07_metric=False,
):
    """Compute VOC AP given precision and recall. If use_07_metric is true, uses
    the VOC 07 11-point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.0
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_ap_score(
    p_bboxes: List[np.ndarray],
    p_scores: List[np.ndarray],
    p_classes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    gt_classes: List[np.ndarray],
    class_id: int = None,
    threshold: float = 0.5,
):
    """
    Args:
        p_bboxes: a list of predict bboxes
        p_scores: a list of predict score for bbox
        p_classes: a list of predict class id for bbox
        gt_bboxes: a list of ground truth bboxes
        gt_classes: a list of true class id for each true bbox
        class_id: the class id to compute ap score
        threshold: the threshold to ap score
    """
    if class_id is not None:
        gt_bboxes = [gt_bbox[gt_class == class_id] for gt_class, gt_bbox in zip(gt_classes, gt_bboxes)]
        p_bboxes = [p_bbox[p_class == class_id] for p_class, p_bbox in zip(p_classes, p_bboxes)]
        p_scores = [p_score[p_class == class_id] for p_class, p_score in zip(p_classes, p_scores)]
        p_indexes = [np.array([i] * len(p_bboxes[i])) for i in range(len(p_bboxes))]

    p_bboxes, p_scores, p_indexes = (
        np.concatenate(p_bboxes),
        np.concatenate(p_scores),
        np.concatenate(p_indexes),
    )
    p_sort_indexes = np.argsort(-p_scores)

    tp = np.zeros(p_scores.shape[0])
    fp = np.zeros(p_scores.shape[0])
    gt_bbox_status = defaultdict(set)
    for idx, p_sort_index in enumerate(p_sort_indexes):
        p_index = int(p_indexes[p_sort_index])
        gt_bbox = gt_bboxes[p_index]
        p_bbox = p_bboxes[p_sort_index]
        vmax = -float("inf")
        jmax = -1
        if gt_bbox.size > 0:
            ixmin = np.maximum(gt_bbox[:, 0], p_bbox[0])
            iymin = np.maximum(gt_bbox[:, 1], p_bbox[1])
            ixmax = np.minimum(gt_bbox[:, 2], p_bbox[2])
            iymax = np.minimum(gt_bbox[:, 3], p_bbox[3])
            iw = np.maximum(ixmax - ixmin + 1.0, 0.0)
            ih = np.maximum(iymax - iymin + 1.0, 0.0)
            inters = iw * ih
            uni = (
                (p_bbox[2] - p_bbox[0] + 1.0) * (p_bbox[3] - p_bbox[1] + 1.0)
                + (gt_bbox[:, 2] - gt_bbox[:, 0] + 1.0) * (gt_bbox[:, 3] - gt_bbox[:, 1] + 1.0)
                - inters
            )
            overlaps = inters / uni
            vmax = np.max(overlaps)
            jmax = np.argmax(overlaps)
        if vmax > threshold:
            if jmax not in gt_bbox_status[p_index]:
                tp[idx] = 1
                gt_bbox_status[p_index].add(jmax)
            else:
                fp[idx] = 1
        else:
            fp[idx] = 1
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    rec = tp / float(sum([len(gt) for gt in gt_bboxes]))
    prec = tp / np.maximum(tp + fp, np.finfo(np.float).eps)
    ap = _voc_ap(rec, prec)
    return ap


def voc_map_score(
    p_bboxes: List[np.ndarray],
    p_scores: List[np.ndarray],
    p_classes: List[np.ndarray],
    gt_bboxes: List[np.ndarray],
    gt_classes: List[np.ndarray],
):
    """
    Args:
        p_bboxes: a list of predict bboxes
        p_scores: a list of predict score for bbox
        p_classes: a list of predict class id for bbox
        gt_bboxes: a list of ground truth bboxes
        gt_classes: a list of true class id for each true bbox
    Returns:
        a avg ap score of all classes in ground truth
    """
    classes = set(list(np.concatenate(gt_classes)))
    ap_scores = dict()
    for thres in range(50, 100, 5):
        ap_scores[thres] = [
            voc_ap_score(
                p_bboxes,
                p_scores,
                p_classes,
                gt_bboxes,
                gt_classes,
                c,
                thres / 100,
            )
            for c in classes
        ]

    mAP = {iou: np.mean(x) for iou, x in ap_scores.items()}
    return np.mean(list(mAP.values()))
