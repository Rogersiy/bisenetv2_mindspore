import numpy as np
import mindspore as ms
from mindspore import ops, nn


def cross_entropy(logits, labels, weight, ignore_label=255):
    n, c = logits.shape
    logits = ops.softmax(logits)
    labels_mask = ops.logical_and(labels != ignore_label, labels >= 0)
    labels = ops.select(labels_mask, labels, ops.zeros_like(labels))
    labels = ops.one_hot(labels, c, ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32))
    labels = labels.astype(logits.dtype)
    weight = weight.reshape((1, -1)).astype(logits.dtype)
    weight = ops.stop_gradient(ops.tile(weight, (n, 1)))
    loss = ops.binary_cross_entropy(logits, labels, weight, reduction='none')
    labels_mask = ops.tile(labels_mask.reshape(n, 1), (1, c))
    loss = loss * labels_mask
    return loss.sum(-1).mean()

logits = ms.Tensor(np.array([[3, 5, 6, 9, 12, 33, 42, 12, 32, 72],
                             [3, 5, 6, 9, 12, 33, 42, 12, 32, 72],
                             [3, 5, 6, 9, 12, 33, 42, 12, 32, 72]]), ms.float32)
labels = ms.Tensor(np.array([1, 5, 255]).astype(np.int32))
print(ops.tile(labels.reshape(1, 3), (2, 1)))
# depth, on_value, off_value = 10, ms.Tensor(1.0, ms.float32), ms.Tensor(0.0, ms.float32)
# labels_onehot = ops.one_hot(labels, depth, on_value, off_value)
# weight = ops.arange(10).astype(ms.float32) * 0.1
#
# ce = nn.CrossEntropyLoss(weight=weight, ignore_index=255, reduction="mean")
#
# loss1 = ce(logits, labels)
# print(loss1)
# loss2 = cross_entropy(logits, labels, weight, 255)
# print(loss2)

