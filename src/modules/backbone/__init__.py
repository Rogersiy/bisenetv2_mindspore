import mindspore as ms
from .hrnet import hrnet_w48, hrnet_w32


def create_backbone(initializer, in_channels=3, pretrained=True, backbone_ckpt=""):
    if initializer == "hrnet_w48":
        net = hrnet_w48(in_channels)
    elif initializer == "hrnet_w32":
        net = hrnet_w32(in_channels)
    else:
        raise ValueError(f"Invalid backbone: {initializer}")
    if pretrained:
        ms.load_checkpoint(backbone_ckpt, net)
        print(f"load {backbone_ckpt}")
    return net