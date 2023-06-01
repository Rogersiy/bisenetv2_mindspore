import copy
import cv2
import numpy as np

__all__ = ['RandomFlip', 'RandomHSV', 'Normalize', 'TransposeImage']


class RandomFlip:
    """Random left_right flip
    Args:
        prob (float): the probability of flipping image
    """

    def __init__(self, prob=0.5):
        self.prob = prob
        if not (isinstance(self.prob, float)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, img, label):
        if np.random.random() < self.prob:
            img = cv2.flip(img, 1)
            label = cv2.flip(label, 1)

        return img, label


class RandomHSV:
    """
    HSV color-space augmentation
    """
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5):
        self.gains = [hgain, sgain, vgain]

    def __call__(self, img, label):
        r = np.random.uniform(-1, 1, 3) * self.gains + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(np.uint8)
        lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
        lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        return img, label


class Normalize:
    """
    Args:
        mean (list): the pixel mean
        std (list): the pixel variance
        is_scale (bool): scale the pixel to [0,1]
        norm_type (str): type in ['mean_std', 'none']
    """
    def __init__(self, mean=None, std=None, is_scale=True, norm_type='mean_std'):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type
        self.input_columns = ['image']

        if not (isinstance(self.is_scale, bool) and self.norm_type in ['mean_std', 'none']):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if self.std and reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, img, label):
        """Normalize the image.
        Operators:
            1.(optional) Scale the pixel to [0,1]
            2.(optional) Each pixel minus mean and is divided by std
        """
        img = img.astype(np.float32)

        if self.is_scale:
            scale = 1.0 / 255.0
            img *= scale

        if self.norm_type == 'mean_std':
            mean = self.mean or img.mean((0, 1))
            mean = np.array(mean)[np.newaxis, np.newaxis, :]
            std = self.std or img.var((0, 1))
            std = np.array(std)[np.newaxis, np.newaxis, :]
            img -= mean
            img /= std
        label = label.astype(np.int32)
        if len(label.shape) == 3:
            label = np.squeeze(label)
        return img, label


class TransposeImage:
    """
    Args:
        bgr2rgb (bool): transpose image channel from BGR to RGB
        hwc2chw (bool): transpose image dim from (h, w, c) to (c, h, w)
    """
    def __init__(self, bgr2rgb=True, hwc2chw=True, consider_poly=False):

        self.bgr2rgb = bgr2rgb
        self.hwc2chw = hwc2chw
        self.input_columns = ['image']

        if not (isinstance(bgr2rgb, bool) and isinstance(hwc2chw, bool)):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, img, label):
        if self.bgr2rgb:
            img = img[:, :, ::-1]
        if self.hwc2chw:
            img = img.transpose(2, 0, 1)
        return img, label
