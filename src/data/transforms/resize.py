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

import numpy as np
import cv2

__all__ = ["Resize", "RandomResizeCrop"]


class Resize:
    def __init__(self, target_size=(320, 640), keep_ratio=True, interp=None, ignore_label=255):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w)
        Args:
            target_size (int|list): image target size
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int, option): the interpolation method
            ignore_label (int): ignore label in label.
        """
        super(Resize, self).__init__()
        self.keep_ratio = keep_ratio
        if isinstance(target_size, int):
            target_size = [target_size, target_size]
        self.target_size = target_size
        self.interp = interp
        self.ignore_label = ignore_label

    def resize_image(self, image, scale):
        im_scale_x, im_scale_y = scale
        interp = self.interp if self.interp else (cv2.INTER_AREA if min(scale) < 1 else cv2.INTER_LINEAR)

        image = cv2.resize(image, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
        resize_h, resize_w = image.shape[:2]
        image = cv2.copyMakeBorder(
            image, 0, self.target_size[0] - resize_h, 0, self.target_size[1] - resize_w, cv2.BORDER_CONSTANT
        )  # top, bottom, left, right
        return image

    def resize_label(self, label, scale):
        im_scale_x, im_scale_y = scale
        label = cv2.resize(label, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=cv2.INTER_NEAREST)
        resize_h, resize_w = label.shape[:2]
        label = cv2.copyMakeBorder(
            label, 0, self.target_size[0] - resize_h, 0, self.target_size[1] - resize_w, self.ignore_label
        )  # top, bottom, left, right
        return label

    def __call__(self, img, label):
        """Resize the image numpy."""
        # apply image
        img_shape = img.shape
        if self.keep_ratio:
            im_scale = min(self.target_size[1] / img_shape[1], self.target_size[0] / img_shape[0])
            im_scale_x, im_scale_y = im_scale, im_scale
        else:
            im_scale_y = self.target_size[0] / img_shape[0]
            im_scale_x = self.target_size[1] / img_shape[1]

        img = self.resize_image(img, [im_scale_x, im_scale_y])
        label = self.resize_label(label, [im_scale_x, im_scale_y])
        return img, label


class RandomResizeCrop:
    def __init__(
        self,
        base_size=2048,
        crop_size=(512, 1024),
        downsample_rate=1,
        scale_factor=16,
        keep_ratio=True,
        interp=None,
        ignore_label=255,
        multi_scale=False,
    ):
        """
        Resize image to target size. if keep_ratio is True,
        resize the image's long side to the maximum of target_size
        if keep_ratio is False, resize the image to target size(h, w).
        Crop resized imgae to crop_size.
        Args:
            base_size (int): image target long side.
            crop_size (list): crop target size.
            downsample_rate (int): label down sample rate.
            scale_factor (int): random resize scale factor.
            keep_ratio (bool): whether keep_ratio or not, default true
            interp (int, option): the interpolation method.
            ignore_label (int): ignore label in label.
            multi_scale (bool): if True, apply multi scale random resize.
        """
        super(RandomResizeCrop, self).__init__()
        self.base_size = base_size
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.downsample_rate = 1.0 / downsample_rate
        self.keep_ratio = keep_ratio
        self.interp = interp
        self.ignore_label = ignore_label
        self.multi_scale = multi_scale

    def multi_scale_aug(self, image, label=None, rand_scale=1.0):
        """Augment feature into different scales."""
        long_size = np.int(self.base_size * rand_scale + 0.5)
        h, w, _ = image.shape
        if h > w:
            new_h = long_size
            new_w = np.int(w * long_size / h + 0.5)
        else:
            new_w = long_size
            new_h = np.int(h * long_size / w + 0.5)

        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if label is not None:
            label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            return image
        return image, label

    def pad_image(self, image, shape, padvalue):
        """Pad an image."""
        pad_image = image.copy()
        h, w = image.shape[:2]
        pad_h = max(shape[0] - h, 0)
        pad_w = max(shape[1] - w, 0)
        if pad_h > 0 or pad_w > 0:
            pad_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=padvalue)
        return pad_image

    def rand_crop(self, image, label):
        """Crop a feature at a random location."""
        image = self.pad_image(image, self.crop_size, (0.0, 0.0, 0.0))
        label = self.pad_image(label, self.crop_size, (self.ignore_label,))

        new_h, new_w = label.shape
        x = np.random.randint(0, new_w - self.crop_size[1]) if new_w - self.crop_size[1] > 0 else 0
        y = np.random.randint(0, new_h - self.crop_size[0]) if new_h - self.crop_size[0] > 0 else 0
        image = image[y : y + self.crop_size[0], x : x + self.crop_size[1]]
        label = label[y : y + self.crop_size[0], x : x + self.crop_size[1]]

        return image, label

    def __call__(self, img, label):
        label = np.squeeze(label)
        if self.multi_scale:
            rand_scale = 0.5 + np.random.randint(0, self.scale_factor) / 10.0
            img, label = self.multi_scale_aug(img, label, rand_scale)

        img, label = self.rand_crop(img, label)
        if self.downsample_rate != 1:
            label = cv2.resize(
                label, None, fx=self.downsample_rate, fy=self.downsample_rate, interpolation=cv2.INTER_NEAREST
            )
        return img, label
