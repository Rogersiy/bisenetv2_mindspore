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

from . import transforms

__all__ = ["create_transform"]


def create_transform(transform):
    """
    Convert the list in the yaml file into a set of data enhancement lists
    the list in yaml file maybe like this:
        single_img_transforms:
            - RandomFlip: {prob: 0.5}
            - RandomHSV: {}
    """
    for k, v in transform.items():
        op_cls = getattr(transforms, k)
        f = op_cls(**v)
    return f
