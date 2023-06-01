import math
from mindspore.common.initializer import HeUniform
from mindspore import ops, nn


class ConvModule(nn.Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False,
                 norm='none',
                 act='none'):
        super(ConvModule, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation,
                                group=groups, pad_mode="pad", padding=padding, has_bias=bias,
                                weight_init=HeUniform(math.sqrt(5))))
        if norm == "bn":
            layers.append(nn.BatchNorm2d(out_channels, eps=1e-4))
        elif norm != 'none':
            raise ValueError(f"not support norm: {norm}, you can set norm None or 'bn'")
        if act != 'none':
            layers.append(nn.get_activation(act))
        self.conv = nn.SequentialCell(layers)

    def construct(self, x):
        return self.conv(x)


class FCNHead(nn.Cell):
    """Fully Convolution Networks for Semantic Segmentation.

    This head is implemented of `FCNNet <https://arxiv.org/abs/1411.4038>`_.

    Args:
        num_convs (int): Number of convs in the head. Default: 2.
        kernel_size (int): The kernel size for convs in the head. Default: 3.
        concat_input (bool): Whether concat the input and output of convs
            before classification layer.
        dilation (int): The dilation rate for convs in the head. Default: 1.
    """

    def __init__(self,
                 in_channels,
                 channels,
                 out_channels=None,
                 num_classes=2,
                 num_convs=2,
                 in_index=-1,
                 kernel_size=3,
                 concat_input=True,
                 dilation=1,
                 norm='none',
                 act='relu',
                 align_corners=None):
        super(FCNHead, self).__init__()
        if isinstance(in_channels, (list, tuple)):
            self.in_channels = sum(in_channels)
        else:
            self.in_channels = in_channels
        self.num_convs = num_convs
        self.concat_input = concat_input
        self.kernel_size = kernel_size
        self.channels = channels
        self.in_index = in_index
        self.align_corners = align_corners

        conv_padding = (kernel_size // 2) * dilation
        convs = []
        convs.append(
            ConvModule(
                self.in_channels,
                self.channels,
                kernel_size=kernel_size,
                padding=conv_padding,
                dilation=dilation,
                norm=norm,
                act=act))
        for i in range(num_convs - 1):
            convs.append(
                ConvModule(
                    self.channels,
                    self.channels,
                    kernel_size=kernel_size,
                    padding=conv_padding,
                    dilation=dilation,
                    norm=norm,
                    act=act))
        if num_convs == 0:
            self.convs = nn.Identity()
        else:
            self.convs = nn.SequentialCell(convs)
        if self.concat_input:
            self.conv_cat = ConvModule(
                self.in_channels + self.channels,
                self.channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                norm=norm,
                act=act)
        if out_channels is None:
            out_channels = num_classes
        self.conv_seg = nn.Conv2d(channels, out_channels, kernel_size=1, weight_init=HeUniform(math.sqrt(5)))

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        upsampled_inputs = ()
        for idx in self.in_index:
            inp = inputs[idx]
            inp = ops.interpolate(inp, size=inputs[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            upsampled_inputs += (inp,)

        inputs = ops.concat(upsampled_inputs, axis=1)
        feats = self.convs(inputs)
        if self.concat_input:
            feats = self.conv_cat(ops.concat((inputs, feats), axis=1))
        return feats

    def construct(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.conv_seg(output)
        return output