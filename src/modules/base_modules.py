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

import math
import mindspore as ms
from mindspore import ops, nn

class MultiScaleInfer(nn.Cell):
    def __init__(self, net, num_classes=2, img_ratios=(1.0,), flip=False, multi_out=True):
        super(MultiScaleInfer, self).__init__()
        self.net = net
        self.num_classes = num_classes
        self.img_ratios = img_ratios
        self.flip = flip
        self.multi_out = multi_out

    def construct(self, img):
        n, c, h, w = img.shape
        pred_res = ops.zeros((n, h, w, self.num_classes), ms.float32)
        for r in self.img_ratios:
            n_h, n_w = int(h * r), int(w * r)
            n_img = ops.interpolate(img, sizes=(n_h, n_w), mode="bilinear")
            pred = self.net(n_img)
            if self.multi_out:
                pred = pred[0]
            pred = ops.interpolate(pred, sizes=(h, w), mode="bilinear")
            pred = ops.softmax(pred.transpose(0, 2, 3, 1), -1)
            pred_res += pred
            if self.flip:
                n_img = n_img[:, :, :, ::-1]
                pred = self.net(n_img)
                if self.multi_out:
                    pred = pred[0]
                pred = pred[:, :, :, ::-1]
                pred = ops.interpolate(pred, sizes=(h, w), mode="bilinear")
                pred = ops.softmax(pred.transpose(0, 2, 3, 1), -1)
                pred_res += pred
        pred_res = ops.argmax(pred_res, -1)
        return pred_res
