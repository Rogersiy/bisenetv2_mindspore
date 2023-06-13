import mindspore as ms
from mindspore import ops, nn


class WithLossCell(nn.Cell):
    r"""
    Cell with loss function.

    Wraps the network with loss function. This Cell accepts data and label as inputs and
    the computed loss will be returned.

    Args:
        backbone (Cell): The backbone network to wrap.
        loss_fn (Cell): The loss function used to compute loss.
        loss_weight (list): weights for each loss_fn

    Inputs:
        - **data** (Tensor) - Tensor of shape :math:`(N, \ldots)`.
        - **label** (Tensor) - Tensor of shape :math:`(N, \ldots)`.

    Outputs:
        Tensor, a tensor means the loss value, the shape of which is usually :math:`()`.
    """

    def __init__(self, net, loss_fn, loss_weight):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.net = net
        self.loss_fn = loss_fn
        self.loss_weight = loss_weight

    def construct(self, data, label):
        logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4 = self.net(data)
        loss = 0
        loss = self.loss_fn(logits, label)
        loss += self.loss_fn(logits_aux2, label)
        loss += self.loss_fn(logits_aux3, label)
        loss += self.loss_fn(logits_aux4, label)
        loss += self.loss_fn(logits_aux5_4, label)
        return loss
    
# class WithLossCell(nn.Cell):
#     def __init__(self, backbone, use_aux=True, lb_ignore=255):
#         super(WithLossCell,self).__init__()
#         self.backbone = backbone
#         self.criteria_pre = OhemCELoss(thresh=0.7,lb_ignore=lb_ignore)
#         self.criteria_aux = OhemCELoss(thresh=0.7,lb_ignore=lb_ignore)
#         self.use_aux = use_aux

#     def construct(self, data, label):
#         logits, logits_aux2, logits_aux3, logits_aux4, logits_aux5_4 = self.backbone(data)
#         # logits = self.backbone(data)
#         loss = self.criteria_pre(logits, label)
#         loss += self.criteria_aux(logits_aux2, label)
#         loss += self.criteria_aux(logits_aux3, label)
#         loss += self.criteria_aux(logits_aux4, label)
#         loss += self.criteria_aux(logits_aux5_4, label)

#         return loss