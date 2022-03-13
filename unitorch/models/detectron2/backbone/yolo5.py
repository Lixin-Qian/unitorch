# Copyright (c) FULIUCANSHENG.
# Licensed under the MIT License.

import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        act=True,
    ):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            p,
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    # Standard bottleneck
    # ch_in, ch_out, shortcut, groups, expansion
    def __init__(
        self,
        c1,
        c2,
        shortcut=True,
        g=1,
        e=0.5,
    ):
        super().__init__()
        hc = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, hc, 1, 1)
        self.cv2 = Conv(hc, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
    ):
        super().__init__()
        hc = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, hc, 1, 1)
        self.cv2 = nn.Conv2d(c1, hc, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(hc, hc, 1, 1, bias=False)
        self.cv4 = Conv(2 * hc, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * hc)  # applied to cat(cv2, cv3)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.m = nn.Sequential(*[Bottleneck(hc, hc, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(
        self,
        c1,
        c2,
        n=1,
        shortcut=True,
        g=1,
        e=0.5,
    ):
        super().__init__()
        hc = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, hc, 1, 1)
        self.cv2 = Conv(c1, hc, 1, 1)
        self.cv3 = Conv(2 * hc, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*[Bottleneck(hc, hc, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        hc = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, hc, 1, 1)
        self.cv2 = Conv(hc * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(
        self,
        c1,
        c2,
        k=1,
        s=1,
        p=None,
        g=1,
        act=True,
    ):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        )


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        (
            N,
            C,
            H,
            W,
        ) = x.size()  # assert (H / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(N, C, H // s, s, W // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(N, C * s * s, H // s, W // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        N, C, H, W = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(N, s, s, C // s ** 2, H, W)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(N, C // s ** 2, H * s, W * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class YOLOV5BackBone(Backbone):
    def __init__(
        self,
        focus=True,
        version="l",
    ):
        super().__init__()
        self.version = version
        self.with_focus = focus
        gains = {
            "s": {"gd": 0.33, "gw": 0.5},
            "m": {"gd": 0.67, "gw": 0.75},
            "l": {"gd": 1, "gw": 1},
            "x": {"gd": 1.33, "gw": 1.25},
        }
        self.gd = gains[self.version]["gd"]  # depth gain
        self.gw = gains[self.version]["gw"]  # width gain

        self.channels_out = {
            "stage1": 64,
            "stage2_1": 128,
            "stage2_2": 128,
            "stage3_1": 256,
            "stage3_2": 256,
            "stage4_1": 512,
            "stage4_2": 512,
            "stage5": 1024,
            "spp": 1024,
            "csp1": 1024,
            "conv1": 512,
            "stage6_1": 512,
            "stage6_2": 512,
            "stage6_3": 256,
            "stage7_1": 256,
            "stage7_2": 256,
            "stage7_3": 256,
            "stage8_1": 512,
            "stage8_2": 512,
            "stage8_3": 512,
            "stage9_1": 1024,
            "stage9_2": 1024,
        }

        for k, v in self.channels_out.items():
            self.channels_out[k] = self.get_width(v)

        if self.with_focus:
            self.stage1 = Focus(3, self.channels_out["stage1"], 3)
        else:
            self.stage1 = Conv(3, self.channels_out["stage1"], 3, 2)

        # for latest yolov5, you can change BottleneckCSP to C3
        self.stage2_1 = Conv(
            self.channels_out["stage1"],
            self.channels_out["stage2_1"],
            k=3,
            s=2,
        )
        self.stage2_2 = C3(
            self.channels_out["stage2_1"],
            self.channels_out["stage2_2"],
            self.get_depth(3),
        )
        self.stage3_1 = Conv(
            self.channels_out["stage2_2"],
            self.channels_out["stage3_1"],
            3,
            2,
        )
        self.stage3_2 = C3(
            self.channels_out["stage3_1"],
            self.channels_out["stage3_2"],
            self.get_depth(9),
        )
        self.stage4_1 = Conv(
            self.channels_out["stage3_2"],
            self.channels_out["stage4_1"],
            3,
            2,
        )
        self.stage4_2 = C3(
            self.channels_out["stage4_1"],
            self.channels_out["stage4_2"],
            self.get_depth(9),
        )
        self.stage5 = Conv(
            self.channels_out["stage4_2"],
            self.channels_out["stage5"],
            3,
            2,
        )
        self.spp = SPP(
            self.channels_out["stage5"],
            self.channels_out["spp"],
            [5, 9, 13],
        )
        self.csp1 = C3(
            self.channels_out["spp"],
            self.channels_out["csp1"],
            self.get_depth(3),
            False,
        )
        self.conv1 = Conv(
            self.channels_out["csp1"],
            self.channels_out["conv1"],
            1,
            1,
        )
        self.upscore1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.stage6_1 = Concat()
        self.stage6_2 = C3(
            self.channels_out["conv1"] + self.channels_out["stage4_2"],
            self.channels_out["stage6_2"],
            self.get_depth(3),
            False,
        )
        self.stage6_3 = Conv(
            self.channels_out["stage6_2"],
            self.channels_out["stage6_3"],
            1,
            1,
        )
        self.upscore2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.stage7_1 = Concat()
        self.stage7_2 = C3(
            self.channels_out["stage6_3"] + self.channels_out["stage3_2"],
            self.channels_out["stage7_2"],
            self.get_depth(3),
            False,
        )
        self.stage7_3 = Conv(
            self.channels_out["stage7_2"],
            self.channels_out["stage7_3"],
            3,
            2,
        )
        self.stage8_1 = Concat()
        self.stage8_2 = C3(
            self.channels_out["stage6_3"] + self.channels_out["stage7_3"],
            self.channels_out["stage8_2"],
            self.get_depth(3),
            False,
        )
        self.stage8_3 = Conv(
            self.channels_out["stage8_2"],
            self.channels_out["stage8_3"],
            3,
            2,
        )
        self.stage9_1 = Concat()
        self.stage9_2 = C3(
            self.channels_out["conv1"] + self.channels_out["stage8_3"],
            self.channels_out["stage9_2"],
            self.get_depth(3),
            False,
        )

        self.out_features = ["p3", "p4", "p5"]

        self.out_feature_channels = {
            "p3": self.channels_out["stage7_2"],
            "p4": self.channels_out["stage8_2"],
            "p5": self.channels_out["stage9_2"],
        }

        self.out_feature_strides = {
            "p3": 1,
            "p4": 1,
            "p5": 1,
        }

    def forward(self, x):
        x = self.stage1(x)
        x21 = self.stage2_1(x)
        x22 = self.stage2_2(x21)
        x31 = self.stage3_1(x22)
        c3 = self.stage3_2(x31)
        x41 = self.stage4_1(c3)
        c4 = self.stage4_2(x41)
        x5 = self.stage5(c4)
        spp = self.spp(x5)
        csp1 = self.csp1(spp)
        c5 = self.conv1(csp1)
        up1 = self.upscore1(c5)
        x61 = self.stage6_1([up1, c4])
        x62 = self.stage6_2(x61)
        c6 = self.stage6_3(x62)
        up2 = self.upscore2(c6)
        x7 = self.stage7_1([up2, c3])
        p3 = self.stage7_2(x7)
        c7 = self.stage7_3(p3)
        x8 = self.stage8_1([c7, c6])
        p4 = self.stage8_2(x8)
        c8 = self.stage8_3(p4)
        x9 = self.stage9_1([c8, c5])
        p5 = self.stage9_2(x9)
        return {"p3": p3, "p4": p4, "p5": p5}

    def get_depth(self, n):
        return max(round(n * self.gd), 1) if n > 1 else n

    def get_width(self, n):
        return make_divisible(n * self.gw, 8)

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self.out_feature_channels[name],
                stride=self.out_feature_strides[name],
            )
            for name in self.out_features
        }


@BACKBONE_REGISTRY.register()
def build_yolo_v5_backbone(cfg, input_shape):
    focus = cfg.MODEL.YOLOV5.FOCUS
    version = cfg.MODEL.YOLOV5.VERSION

    model = YOLOV5BackBone(
        focus=focus,
        version=version,
    )

    return model
