import mindspore as ms
from mindspore import ops, nn


class CrossEntropy(nn.LossBase):
    """
    Cross-entropy loss function for semantic segmentation,
    and different classes have different weights.
    """

    def __init__(self, num_classes=19, ignore_label=255, cls_weight=None):
        super(CrossEntropy, self).__init__()
        weight = ms.Tensor([1.0] * num_classes, ms.float32)
        self.ignore_label = ignore_label
        if cls_weight is not None:
            weight = ms.Tensor(cls_weight, ms.float32)
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_label, reduction='mean', label_smoothing=0.1)

    def construct(self, logits, labels):
        """Loss construction."""
        labels = labels.astype(ms.int32).reshape((-1,))
        labels = ops.select(labels >= 0, labels, ops.ones_like(labels) * self.ignore_label)
        logits = ops.transpose(logits, (0, 2, 3, 1)).reshape((-1, self.num_classes))
        loss = self.ce(logits, labels)
        return loss