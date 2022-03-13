# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import logging
import math
import time
import numpy as np
import torch
import torchvision
from torch import nn, Tensor
from typing import List, Dict, Tuple

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, Conv2d
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.backbone import Backbone
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling import build_backbone


class YoloV5Head(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        input_shape: List[ShapeSpec],
        nc,
        anchors,
    ):

        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        assert self.nl == len(input_shape)
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer("anchors", a)  # shape(nl,na,2)
        self.register_buffer("anchor_grid", a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        ch = [x.channels for x in input_shape]
        self.m = nn.ModuleList(Conv2d(x, self.no * self.na, 1) for x in ch)

    @classmethod
    def from_config(cls, cfg, input_shape: List[ShapeSpec]):
        nc = cfg.MODEL.YOLOV5.NUM_CLASSES
        anchors = cfg.MODEL.YOLOV5.ANCHORS
        return {
            "input_shape": input_shape,
            "nc": nc,
            "anchors": anchors,
        }

    def forward(self, x: List[Tensor]):
        """
        Arguments:
            features (list[Tensor]): FPN feature map tensors in high to low resolution.
                Each tensor in the list correspond to different feature levels.
        Returns:
            x (list[Tensor]): #nl tensors,
                                each having shape [N, na, Hi, Wi, nc + 5]
            z (Tensor) : [N, nl*na*(sum of grid sizes) , no] indictaing
                    1. Box position z[..., 0:2]
                    2. Box width and height z[..., 2:4]
                    3. Objectness z[..., 5]
                    4. Class probabilities z[..., 6:]
        """
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        return x

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        for mi, s in zip(self.m, self.stride):  # from
            b = mi.bias.view(self.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (self.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
):
    """Runs Non-Maximum Suppression (NMS) on inference results
        conf_thresh - 0.1 (for yolov4)
        iou_thresh - 0.6 (for yolov4)
        multi_label - not in yolov4
        merge = False - in yolov4 not in yolov3
        Labesl = () not in yolov4
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # if yolov3 or yolov5:
    nc = prediction.shape[2] - 5  # number of classes
    # else
    nc = prediction[0].shape[1] - 5  # Number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # (pixels) minimum and maximum box width and height
    _, max_wh = 2, 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling - Not used in the YOLOV4
        if labels and len(labels[xi]):
            l_ = labels[xi]
            v = torch.zeros((len(l_), nc + 5), device=x.device)
            v[:, :4] = l_[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l_)), l_[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)
        #################################################################
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        # #### Not in Yolov4 ######################
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]
        ###############################################
        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f"WARNING: NMS time limit {time_limit}s exceeded")
            break  # time limit exceeded

    return output


def bbox_iou(
    box1,
    box2,
    x1y1x2y2=True,
    GIoU=False,
    DIoU=False,
    CIoU=False,
    EIoU=False,
    ECIoU=False,
    eps=1e-7,
):
    # eps default value used in yolov4 is 1e-9
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * (
        torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)
    ).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU or EIoU or ECIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU or EIoU or ECIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = (
                (b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2
            ) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU

            # ################### Function from Yolov4 ###########################################
            elif EIoU:  # Efficient IoU https://arxiv.org/abs/2101.08158
                rho3 = (w1 - w2) ** 2
                c3 = cw ** 2 + eps
                rho4 = (h1 - h2) ** 2
                c4 = ch ** 2 + eps
                return iou - rho2 / c2 - rho3 / c3 - rho4 / c4  # EIoU
            elif ECIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                rho3 = (w1 - w2) ** 2
                c3 = cw ** 2 + eps
                rho4 = (h1 - h2) ** 2
                c4 = ch ** 2 + eps
                return iou - v * alpha - rho2 / c2 - rho3 / c3 - rho4 / c4  # ECIoU
            ############################################################################################
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU


def smooth_BCE(
    eps=0.1,
):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria =
    # FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power
        # for gradient stability

        # TF implementation
        # https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss(object):
    # Compute losses

    @configurable
    def __init__(
        self,
        *,
        focal_loss_gamma,
        box_loss_gain,
        cls_loss_gain,
        cls_positive_weight,
        obj_loss_gain,
        obj_positive_weight,
        label_smoothing=0.0,
        gr,
        na,
        nc,
        nl,
        anchors,
        anchor_t,
        autobalance=False,
    ):
        super().__init__()
        self.sort_obj_iou = False
        self.na = na
        self.nc = nc
        self.nl = nl
        self.anchors = anchors
        self.box_loss_gain = box_loss_gain
        self.cls_loss_gain = cls_loss_gain
        self.obj_loss_gain = obj_loss_gain
        self.anchor_t = anchor_t

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cls_positive_weight]))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([obj_positive_weight]))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # positive, negative BCE targets
        self.cp, self.cn = smooth_BCE(eps=label_smoothing)

        # Focal loss
        if focal_loss_gamma > 0:
            BCEcls = FocalLoss(BCEcls, focal_loss_gamma)
            BCEobj = FocalLoss(BCEobj, focal_loss_gamma)

        # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.autobalance = (
            BCEcls,
            BCEobj,
            gr,
            autobalance,
        )

    @classmethod
    def from_config(cls, cfg, head):
        return {
            "focal_loss_gamma": cfg.MODEL.YOLOV5.FOCAL_LOSS_GAMMA,
            "box_loss_gain": cfg.MODEL.YOLOV5.BOX_LOSS_GAIN,
            "cls_loss_gain": cfg.MODEL.YOLOV5.CLS_LOSS_GAIN,
            "cls_positive_weight": cfg.MODEL.YOLOV5.CLS_POSITIVE_WEIGHT,
            "obj_loss_gain": cfg.MODEL.YOLOV5.OBJ_LOSS_GAIN,
            "obj_positive_weight": cfg.MODEL.YOLOV5.OBJ_POSITIVE_WEIGHT,
            "label_smoothing": cfg.MODEL.YOLOV5.LABEL_SMOOTHING,
            "gr": 1.0,
            "na": head.na,
            "nc": head.nc,
            "nl": head.nl,
            "anchors": head.anchors,
            "anchor_t": cfg.MODEL.YOLOV5.ANCHOR_T,
            "autobalance": False,
        }

    def _initialize_ssi(self, stride):
        if self.autobalance:
            self.ssi = list(stride).index(16)

    def __call__(self, p, instances):  # predictions, targets, model is ignored
        device = instances[0].gt_boxes.device
        self.to(device)
        lcls, lbox, lobj = (
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
        )
        tcls, tbox, indices, anchors = self.build_targets(p, instances)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # prediction subset corresponding to targets
                ps = pi[b, a, gj, gi]

                # Regression
                pxy = ps[:, :2].sigmoid() * 2.0 - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                # iou(prediction, target)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()  # iou loss

                # Objectness
                score_iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    sort_id = torch.argsort(score_iou)
                    b, a, gj, gi, score_iou = (
                        b[sort_id],
                        a[sort_id],
                        gj[sort_id],
                        gi[sort_id],
                        score_iou[sort_id],
                    )
                tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(ps[:, 5:], t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.box_loss_gain
        lobj *= self.obj_loss_gain
        lcls *= self.cls_loss_gain
        # bs = tobj.shape[0]  # batch size

        # loss = lbox + lobj + lcls
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        return {
            "loss_box": lbox,
            "loss_obj": lobj,
            "loss_cls": lcls,
        }

    def build_targets(self, p, gt_instances):
        """
        Args:
            p (list[Tensors]): A list of #feature level predictions
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.
        """
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        targets = []
        for i, gt_per_image in enumerate(gt_instances):
            # Convert the boxes to target format of shape [sum(nL per image), 6]
            # where each target entry is [img_index, class, x, y, w, h],
            # x, y, w, h - relative and x, y are centers
            if len(gt_per_image) > 0:
                boxes = gt_per_image.gt_boxes.tensor.clone()
                h, w = gt_per_image.image_size
                boxes[:, 0:2] = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
                boxes[:, 2:4] = (boxes[:, 2:4] - boxes[:, 0:2]) * 2
                boxes[:, ::2] /= float(w)
                boxes[:, 1::2] /= float(h)
                classes = torch.unsqueeze(gt_per_image.gt_classes.clone(), dim=1)
                t = torch.cat([torch.ones_like(classes) * i, classes, boxes], dim=1)
                targets.append(t)
        targets = torch.cat(targets, 0)

        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        # normalized to gridspace gain
        gain = torch.ones(7, device=targets.device)
        ai = (
            torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)
        )  # same as .repeat_interleave(nt)
        # append anchor indices
        targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)

        g = 0.5  # bias
        off = (
            torch.tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],  # j,k,l,m
                    # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                ],
                device=targets.device,
            ).float()
            * g
        )  # offsets

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if nt:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1.0 / r).max(2)[0] < self.anchor_t  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  #
                # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1.0 < g) & (gxy > 1.0)).T
                l, m = ((gxi % 1.0 < g) & (gxi > 1.0)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T  # image, class
            gxy = t[:, 2:4]  # grid xy
            gwh = t[:, 4:6]  # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            # image, anchor, grid indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch

    def to(self, device):
        self.anchors = self.anchors.to(device)
        self.BCEcls.pos_weight = self.BCEcls.pos_weight.to(device)
        self.BCEobj.pos_weight = self.BCEobj.pos_weight.to(device)


@META_ARCH_REGISTRY.register()
class YoloV5(nn.Module):
    """
    Implement YoloV5
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        head: nn.Module,
        loss,
        num_classes,
        conf_thres,
        iou_thres,
        pixel_mean,
        pixel_std,
        vis_period=0,
        input_format="BGR",
    ):
        super().__init__()

        self.backbone = backbone
        self.head = head

        self.num_classes = num_classes
        self.single_cls = num_classes == 1
        # Inference Parameters
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        # Vis parameters
        self.vis_period = vis_period
        self.input_format = input_format

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

        """
        In Detectron1, loss is normalized by number of foreground samples in the batch.
        When batch size is 1 per GPU, #foreground has a large variance and
        using it lead to lower performance. Here we maintain an EMA of #foreground to
        stabilize the normalizer.
        """
        self.loss = loss
        # self.loss_normalizer = 100  # initialize with any reasonable #fg that's not too small
        # self.loss_normalizer_momentum = 0.9
        self.init_stride()
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.BatchNorm2d):
            module.eps = 1e-3
            module.momentum = 0.03

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = list(backbone_shape.values())
        head = YoloV5Head(cfg, feature_shapes)
        loss = ComputeLoss(cfg, head)
        return {
            "backbone": backbone,
            "head": head,
            "loss": loss,
            "num_classes": head.nc,
            "conf_thres": cfg.MODEL.YOLOV5.CONF_THRESH,
            "iou_thres": cfg.MODEL.YOLOV5.IOU_THRES,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "vis_period": cfg.VIS_PERIOD,
            "input_format": cfg.INPUT.FORMAT,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, results):
        """
        A function used to visualize ground truth images and final network predictions.
        It shows ground truth bounding boxes on the original image and up to 20
        predicted object bounding boxes on the original image.
        Args:
            batched_inputs (list): a list that contains input to the model.
            results (List[Instances]): a list of #images elements.
        """
        from detectron2.utils.visualizer import Visualizer

        assert len(batched_inputs) == len(results), "Cannot visualize inputs and results of different sizes"
        storage = get_event_storage()
        max_boxes = 20

        image_index = 0  # only visualize a single image
        img = batched_inputs[image_index]["image"]
        img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
        v_gt = Visualizer(img, None)
        v_gt = v_gt.overlay_instances(boxes=batched_inputs[image_index]["instances"].gt_boxes)
        anno_img = v_gt.get_image()
        processed_results = detector_postprocess(results[image_index], img.shape[0], img.shape[1])
        predicted_boxes = processed_results.pred_boxes.tensor.detach().cpu().numpy()

        v_pred = Visualizer(img, None)
        v_pred = v_pred.overlay_instances(boxes=predicted_boxes[0:max_boxes])
        prop_img = v_pred.get_image()
        vis_img = np.vstack((anno_img, prop_img))
        vis_img = vis_img.transpose(2, 0, 1)
        vis_name = f"Top: GT bounding boxes; Bottom: {max_boxes} Highest Scoring Results"
        storage.put_image(vis_name, vis_img)

    def init_stride(self):
        s = 256  # 2x min stride
        dummy_input = torch.zeros(1, len(self.pixel_mean), s, s)
        features = self.backbone(dummy_input)
        features = list(features.values())
        pred = self.head(features)
        self.head.stride = torch.tensor([s / x.shape[-2] for x in pred])  # forward
        self.head.anchors /= self.head.stride.view(-1, 1, 1)
        self.stride = self.head.stride
        self.head._initialize_biases()  # only run once
        self.loss._initialize_ssi(self.stride)

    def forward(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                * image: Tensor, image in (C, H, W) format.
                * instances: Instances
                Other information that's included in the original dicts, such as:
                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.
        Returns:
            In training, dict[str, Tensor]: mapping from a named loss to a tensor storing the
            loss. Used during training only. In inference, the standard output format, described
            in :doc:`/tutorials/models`.
        """
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = list(features.values())

        pred = self.head(features)

        if self.training:
            assert not torch.jit.is_scripting(), "Not supported"
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            losses = self.loss(pred, gt_instances)

            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    results = self.inference(pred, images.image_sizes)
                    self.visualize_training(batched_inputs, results)

            return losses
        else:
            results = self.inference(pred, images.image_sizes)
            if torch.jit.is_scripting():
                return results
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def inference(self, x, image_sizes):
        """
        Returns:
        z (Tensor) : [N, nl*na*(sum of grid sizes) , no] indictaing
                    1. Box position z[..., 0:2]
                    2. Box width and height z[..., 2:4]
                    3. Objectness z[..., 5]
                    4. Class probabilities z[..., 6:]
        """
        z = []
        for i in range(self.head.nl):
            # x(bs,na,ny,nx,no)
            bs, _, ny, nx, _ = x[i].shape
            if self.head.grid[i].shape[2:4] != x[i].shape[2:4]:
                self.head.grid[i] = self.head._make_grid(nx, ny).to(x[i].device)

            y = x[i].sigmoid()
            # if self.head.inplace:
            y[..., 0:2] = (y[..., 0:2] * 2.0 - 0.5 + self.head.grid[i]) * self.head.stride[i]  # xy
            y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.head.anchor_grid[i]  # wh
            # else:  # for YOLOv5 on AWS Inferentia https://github.com/ultralytics/yolov5/pull/2953
            #     xy = (y[..., 0:2] * 2. - 0.5 + self.head.grid[i]) * self.head.stride[i]  # xy
            #     wh = (y[..., 2:4] * 2) ** 2 * self.head.anchor_grid[i].view(1, self.head.na, 1, 1, 2)  # wh
            #     y = torch.cat((xy, wh, y[..., 4:]), -1)
            z.append(y.view(bs, -1, self.head.no))
        return self.process_inference(torch.cat(z, 1), image_sizes)

    def process_inference(self, out, image_sizes):
        out = non_max_suppression(
            out,
            self.conf_thres,
            self.iou_thres,
            multi_label=True,
            agnostic=self.single_cls,
        )
        assert len(out) == len(image_sizes)
        results_all: List[Instances] = []
        # Statistics per image
        for si, (pred, img_size) in enumerate(zip(out, image_sizes)):

            if len(pred) == 0:
                result = Instances(img_size)
                result.pred_boxes = Boxes(torch.tensor([]))
                result.scores = torch.tensor([])
                result.pred_classes = torch.tensor([])
            else:
                # Predictions
                if self.single_cls:
                    pred[:, 5] = 0
                predn = pred.clone()
                # Predn shape [ndets, 6] of format [xyxy, conf, cls] relative to the input image size
                result = Instances(img_size)
                result.pred_boxes = Boxes(predn[:, :4])  # TODO: Check if resizing needed
                result.scores = predn[:, 4]
                result.pred_classes = predn[:, 5].int()  # TODO: Check the classes
            results_all.append(result)
        return results_all

    def preprocess_image(self, batched_inputs: Tuple[Dict[str, Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images
