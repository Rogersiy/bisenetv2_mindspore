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
from mindspore import dataset as ds


def citycapes_dataset(dataset_dir, map_label=True, ignore_label=255, group_size=1, rank=0, is_train=True):
    """
    Cityscapes dataset.
    Args:
        dataset_dir (str): Path to the root directory that contains the dataset.
        map_label (bool): cityscapes dataset default class number is 34, if map_label is True,
                          some classes with a small number will be set to ignore_label,
                          after map_label, class num is 19
        ignore_label (int): ignore label in dataset
        group_size (int): Number of shards that the dataset will be divided
        rank (int): The shard ID within `group_size`
        is_train (bool): whether is training.

    .. code-block::

        .
        └── Cityscapes
             ├── leftImg8bit
             |    ├── train
             |    |    ├── aachen
             |    |    |    ├── aachen_000000_000019_leftImg8bit.png
             |    |    |    ├── aachen_000001_000019_leftImg8bit.png
             |    |    |    ├── ...
             |    |    ├── bochum
             |    |    |    ├── ...
             |    |    ├── ...
             |    ├── test
             |    |    ├── ...
             |    ├── val
             |    |    ├── ...
             └── gtFine
                  ├── train
                  |    ├── aachen
                  |    |    ├── aachen_000000_000019_gtFine_color.png
                  |    |    ├── aachen_000000_000019_gtFine_instanceIds.png
                  |    |    ├── aachen_000000_000019_gtFine_labelIds.png
                  |    |    ├── aachen_000000_000019_gtFine_polygons.json
                  |    |    ├── aachen_000001_000019_gtFine_color.png
                  |    |    ├── aachen_000001_000019_gtFine_instanceIds.png
                  |    |    ├── aachen_000001_000019_gtFine_labelIds.png
                  |    |    ├── aachen_000001_000019_gtFine_polygons.json
                  |    |    ├── ...
                  |    ├── bochum
                  |    |    ├── ...
                  |    ├── ...
                  ├── test
                  |    ├── ...
                  └── val
                       ├── ...

    Citation:

    .. code-block::

        @inproceedings{Cordts2016Cityscapes,
        title       = {The Cityscapes Dataset for Semantic Urban Scene Understanding},
        author      = {Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler,
                        Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
        booktitle   = {Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year        = {2016}
        }
    """
    usage = "train" if is_train else "val"
    dataset = ds.CityscapesDataset(
        dataset_dir=dataset_dir,
        usage=usage,
        task="semantic",
        decode=True,
        num_shards=group_size,
        shard_id=rank,
        shuffle=is_train,
    )
    if map_label:

        def convert_label(image, label, inverse=False):
            """Convert classification ids in labels."""
            if len(label.shape) == 3:
                label = label[:, :, 0]
            temp = label.copy()
            if inverse:
                for v, k in label_mapping.items():
                    label[temp == k] = v
            else:
                for k, v in label_mapping.items():
                    label[temp == k] = v
            image_shape = np.array(image.shape[:2])
            return image, image_shape, label

        label_mapping = {-1: ignore_label, 0: ignore_label,
                         1: ignore_label, 2: ignore_label,
                         3: ignore_label, 4: ignore_label,
                         5: ignore_label, 6: ignore_label,
                         7: 0, 8: 1, 9: ignore_label,
                         10: ignore_label, 11: 2, 12: 3,
                         13: 4, 14: ignore_label, 15: ignore_label,
                         16: ignore_label, 17: 5, 18: ignore_label,
                         19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                         25: 12, 26: 13, 27: 14, 28: 15,
                         29: ignore_label, 30: ignore_label,
                         31: 16, 32: 17, 33: 18}
        dataset = dataset.map(operations=convert_label,
                              input_columns=["image", "task"],
                              output_columns=["image", "ori_shape", "label"],
                              python_multiprocessing=True, max_rowsize=64)
    return dataset
