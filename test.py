import numpy as np
import mindspore as ms
from mindspore import ops, nn

def nms(bbox, threshold):
    box = bbox[:, :4]
    n = box.shape[0]
    iou_mask = ops.iou(box, box) < threshold
    range = ms.numpy.arange(n)
    metri = ops.tile(range.reshape((1, n)), (n, 1))
    mask = metri > range.reshape(n,1)
    print(ops.iou(box, box))
    print(iou_mask)
    print(mask)
    mask = ops.logical_and(mask, iou_mask).astype(range.dtype)
    return ops.reduce_sum(mask, 0) == range

xy = np.random.uniform(0, 20, (10, 2))
wh = np.random.uniform(0, 10, (10, 2))

bbox = ms.Tensor(np.concatenate((xy, xy + wh), -1), ms.float32)
print(bbox)
print(nms(bbox, 0.05))
