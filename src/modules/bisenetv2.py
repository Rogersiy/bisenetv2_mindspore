# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import os.path
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer


class ConvBNReLU(nn.Cell):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1,
                 dilation=1, groups=1, has_bias=False):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_chan,
            out_channels=out_chan, 
            kernel_size=ks,
            stride=stride, 
            pad_mode='pad', 
            padding=padding, 
            dilation=dilation,
            group=groups, 
            has_bias=has_bias)
        self.bn = nn.BatchNorm2d(num_features=out_chan, eps=1e-3)
        self.relu = nn.ReLU()

    def construct(self, x):
        feat = self.conv(x)
        feat = self.bn(feat)
        feat = self.relu(feat)
        return feat


class UpSample(nn.Cell):
    def __init__(self, scale_factor):
        super(UpSample, self).__init__()
        self.scale_factor = scale_factor
    
    def construct(self, x):
        sf = self.scale_factor
        size=(x.shape[2] * sf,x.shape[3] * sf)
        return ops.interpolate(x, sizes=size, mode="bilinear")

        
class DetailBranch(nn.Cell):
    def __init__(self):
        super(DetailBranch, self).__init__()
        self.S1 = nn.SequentialCell(
            ConvBNReLU(3, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S2 = nn.SequentialCell(
            ConvBNReLU(64, 64, 3, stride=2),
            ConvBNReLU(64, 64, 3, stride=1),
            ConvBNReLU(64, 64, 3, stride=1),
        )
        self.S3 = nn.SequentialCell(
            ConvBNReLU(64, 128, 3, stride=2),
            ConvBNReLU(128, 128, 3, stride=1),
            ConvBNReLU(128, 128, 3, stride=1),
        )

    def construct(self, x):
        feat = self.S1(x)
        feat = self.S2(feat)
        feat = self.S3(feat)
        return feat


class StemBlock(nn.Cell):
    def __init__(self):
        super(StemBlock, self).__init__()
        self.conv = ConvBNReLU(3, 16, 3, stride=2)
        self.left = nn.SequentialCell(
            ConvBNReLU(16, 8, 1, stride=1, padding=0),
            ConvBNReLU(8, 16, 3, stride=2),
        )
        self.right=nn.SequentialCell(
            nn.Pad(paddings=((0,0),(0,0),(1,0),(1,0))),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )
        self.fuse = ConvBNReLU(32, 16, 3, stride=1)

    def construct(self, x):
        feat = self.conv(x)
        feat_left = self.left(feat)
        feat_right = self.right(feat)
        feat = ops.concat([feat_left, feat_right], 1)
        feat = self.fuse(feat)
        return feat      


class CEBlock(nn.Cell):

    def __init__(self):
        super(CEBlock, self).__init__()
        self.bn = nn.BatchNorm2d(128, eps=1e-3)
        self.conv_gap = ConvBNReLU(128, 128, 1, stride=1, padding=0)
        #TODO: in paper here is naive conv2d, no bn-relu
        self.conv_last = ConvBNReLU(128, 128, 3, stride=1)

    def construct(self, x):
        feat = ops.mean(x, (2, 3), keep_dims=True)
        feat = self.bn(feat)
        feat = self.conv_gap(feat)
        feat = feat + x
        feat = self.conv_last(feat)
        return feat    
    
    
class GELayerS1(nn.Cell):
    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS1, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv = nn.SequentialCell(
            nn.Conv2d(
                in_chan, 
                mid_chan, 
                kernel_size=3, 
                stride=1,
                padding=1, 
                pad_mode='pad', 
                group=in_chan, 
                has_bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(), # not shown in paper
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan, 
                out_chan, 
                kernel_size=1, 
                stride=1,
                padding=0, 
                pad_mode='pad',
                has_bias=False),
            nn.BatchNorm2d(out_chan, gamma_init='zeros'),
        )
        self.conv2[1].last_bn = True
        self.relu = nn.ReLU()

    def construct(self, x):
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.conv2(feat)
        feat = (feat + x) / 2
        feat = self.relu(feat)
        return feat


class GELayerS2(nn.Cell):

    def __init__(self, in_chan, out_chan, exp_ratio=6):
        super(GELayerS2, self).__init__()
        mid_chan = in_chan * exp_ratio
        self.conv1 = ConvBNReLU(in_chan, in_chan, 3, stride=1)
        self.dwconv1 = nn.SequentialCell(
            nn.Conv2d(
                in_chan, 
                mid_chan, 
                kernel_size=3, 
                stride=2,
                padding=1, 
                pad_mode='pad', 
                group=in_chan, 
                has_bias=False),
            nn.BatchNorm2d(mid_chan),
        )
        self.dwconv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan, 
                mid_chan, 
                kernel_size=3, 
                stride=1,
                padding=1, 
                pad_mode='pad',
                group=mid_chan, 
                has_bias=False),
            nn.BatchNorm2d(mid_chan),
            nn.ReLU(), # not shown in paper
        )
        self.conv2 = nn.SequentialCell(
            nn.Conv2d(
                mid_chan, 
                out_chan, 
                kernel_size=1, 
                stride=1,
                padding=0,
                pad_mode='pad',
                has_bias=False),
            nn.BatchNorm2d(out_chan, gamma_init='zeros'),
        )
        self.conv2[1].last_bn = True
        self.shortcut = nn.SequentialCell(
                nn.Conv2d(
                    in_chan, in_chan, kernel_size=3, stride=2,
                    padding=1, pad_mode='pad', group=in_chan, has_bias=False),
                nn.BatchNorm2d(in_chan),
                nn.Conv2d(
                    in_chan, out_chan, kernel_size=1, stride=1,
                    padding=0, has_bias=False),
                nn.BatchNorm2d(out_chan),
        )
        self.relu = nn.ReLU()

    def construct(self, x):
        feat = self.conv1(x)
        feat = self.dwconv1(feat)
        feat = self.dwconv2(feat)
        feat = self.conv2(feat)
        shortcut = self.shortcut(x)
        feat = feat + shortcut
        feat = self.relu(feat)
        return feat


class SegmentBranch(nn.Cell):

    def __init__(self):
        super(SegmentBranch, self).__init__()
        self.S1S2 = StemBlock()
        self.S3 = nn.SequentialCell(
            GELayerS2(16, 32),
            GELayerS1(32, 32),
        )
        self.S4 = nn.SequentialCell(
            GELayerS2(32, 64),
            GELayerS1(64, 64),
        )
        self.S5_4 = nn.SequentialCell(
            GELayerS2(64, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
            GELayerS1(128, 128),
        )
        self.S5_5 = CEBlock()

    def construct(self, x):
        feat2 = self.S1S2(x)
        feat3 = self.S3(feat2)
        feat4 = self.S4(feat3)
        feat5_4 = self.S5_4(feat4)
        feat5_5 = self.S5_5(feat5_4)
        return feat2, feat3, feat4, feat5_4, feat5_5


class BGALayer(nn.Cell):

    def __init__(self):
        super(BGALayer, self).__init__()
        self.left1 = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, pad_mode='pad', group=128, has_bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, has_bias=False),
        )
        self.left2 = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=2,
                padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=3, stride=2, pad_mode='same')
        )
        self.right1 = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(128),
        )
        self.right2 = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, pad_mode='pad', group=128, has_bias=False),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 128, kernel_size=1, stride=1,
                padding=0, has_bias=False),
        )
        self.up1 = UpSample(scale_factor=4) # (scale_factor=4)
        self.up2 = UpSample(scale_factor=4)
        self.conv = nn.SequentialCell(
            nn.Conv2d(
                128, 128, kernel_size=3, stride=1,
                padding=1, pad_mode='pad', has_bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(), # not shown in paper
        )

    def construct(self, x_d, x_s):
        #dsize = x_d.size[2:]
        left1 = self.left1(x_d)
        left2 = self.left2(x_d)
        right1 = self.right1(x_s)
        right2 = self.right2(x_s)
        right1 = self.up1(right1)

        left = left1 * ops.sigmoid(right1)
        right = left2 * ops.sigmoid(right2)
        right = self.up2(right)
        out = self.conv(left + right)
        return out


class UpSampleBilinear(nn.Cell):

    def __init__(self, scale_factor, align_corners=False):
        super(UpSampleBilinear, self).__init__()
        self.scale_factor = int(scale_factor)
        self.align_corners=align_corners


    def construct(self, x):
        h, w = x.shape[-2:]
        coordinate_mode='half_pixel'
        if self.align_corners == True:
             coordinate_mode='align_corners'
        result = ops.interpolate(x, sizes=(h * self.scale_factor, w * self.scale_factor),
                                 coordinate_transformation_mode=coordinate_mode,mode="bilinear")
        return result



class SegmentHead(nn.Cell):
    def __init__(self, in_chan, mid_chan, n_classes, up_factor=8, aux=True):
        super(SegmentHead, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, 3, stride=1)
        #self.drop = nn.Dropout(keep_prob=0.9)
        self.up_factor = up_factor

        out_chan = n_classes
        mid_chan2 = up_factor * up_factor if aux else mid_chan
        up_factor = up_factor // 2 if aux else up_factor
        # self.up_factorex=up_factor
        if aux:
            self.conv_out = nn.SequentialCell([
                UpSample(scale_factor=2),
                ConvBNReLU(mid_chan, mid_chan2, 3, stride=1),
                nn.Conv2d(mid_chan2, out_chan, 1, 1, padding=0,pad_mode='pad', has_bias=True),
                UpSampleBilinear(scale_factor=up_factor,align_corners=False)
            ])
        else:
            self.conv_out = nn.SequentialCell([
                nn.Conv2d(mid_chan2, out_chan, 1, 1, padding=0, pad_mode='pad', has_bias=True),
                UpSampleBilinear(scale_factor=up_factor, align_corners=False)
            ])


    def construct(self, x):
        feat = self.conv(x)
        # feat = self.drop(feat)
        feat = self.conv_out(feat)
        return feat


class BiSeNetV2(nn.Cell):

    def __init__(self, n_classes, aux_mode='train', backbone_url="backbone_v2.ckpt"):
        super(BiSeNetV2, self).__init__()
        self.aux_mode = aux_mode
        self.detail = DetailBranch()
        self.segment = SegmentBranch()
        self.bga = BGALayer()

        self.head = SegmentHead(128, 1024, n_classes, up_factor=8, aux=False)
        if self.aux_mode == 'train':
            self.aux2 = SegmentHead(16, 128, n_classes, up_factor=4)
            self.aux3 = SegmentHead(32, 128, n_classes, up_factor=8)
            self.aux4 = SegmentHead(64, 128, n_classes, up_factor=16)
            self.aux5_4 = SegmentHead(128, 128, n_classes, up_factor=32)
        self.backbone_url = backbone_url
        self.init_weights()

    def construct(self, x):

        feat_d = self.detail(x)
        feat2, feat3, feat4, feat5_4, feat_s = self.segment(x)
        feat_head = self.bga(feat_d, feat_s)
        logits = self.head(feat_head)
        if self.aux_mode == 'train':
            logits_aux2 = self.aux2(feat2)
            logits_aux3 = self.aux3(feat3)
            logits_aux4 = self.aux4(feat4)
            logits_aux5_4 = self.aux5_4(feat5_4)
            return logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4
        elif self.aux_mode == 'eval':
            return logits
        else:
            return 0

    def init_weights(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, (nn.Conv2d, nn.Dense)):
                cell.weight.set_data(ms.common.initializer.initializer(
                    ms.common.initializer.HeUniform(),
                    cell.weight.shape, cell.weight.dtype))


    def get_params(self):
        def add_param_to_list(mod, wd_params, nowd_params):
            for param in mod.trainable_params():
                if param.dim() == 1:
                    nowd_params.append(param)
                elif param.dim() == 4:
                    wd_params.append(param)
                else:
                    print(name)

        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.name_cells().items():
            if 'head' in name or 'aux' in name:
                add_param_to_list(child, lr_mul_wd_params, lr_mul_nowd_params)
            else:
                add_param_to_list(child, wd_params, nowd_params)
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params
