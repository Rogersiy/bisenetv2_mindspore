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

import mindspore as ms
from mindspore import ops
import numpy as np


def get_confusion_matrix(label, pred, num_class, ignore=-1, rank_size=1):
    """
    Calcute the confusion matrix by given label and pred.
    """
    seg_gt = label.astype(dtype=np.int32)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype("int32")
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred] = label_count[cur_index]
    if rank_size > 1:
        confusion_matrix = ms.Tensor(confusion_matrix, ms.float32)
        allreduce_sum = ops.AllReduce(ops.ReduceOp.SUM)
        confusion_matrix = allreduce_sum(confusion_matrix).asnumpy()
    return confusion_matrix
