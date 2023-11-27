#!/bin/bash
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

if [ $# != 3 ]
then
    echo "Using: bash scripts/run_distribute_train.sh [DEVICE_NUM] [BATCH_SIZE] [SAVE_DIR]"
    exit 1
fi


DEVICE_NUM=$1
echo $DEVICE_NUM



BATCH_SIZE=$2
echo $BATCH_SIZE

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

SAVE_DIR=$(get_real_path $3)

if [ -d ${SAVE_DIR} ]; then
  rm -rf ${SAVE_DIR}
fi
mkdir -p ${SAVE_DIR}


env >${SAVE_DIR}/env.log

mpirun --allow-run-as-root -n $DEVICE_NUM --merge-stderr-to-stdout \
    python train.py --config config/bisenetv2/config_bisenetv2_16k.yml --batch_size $BATCH_SIZE --device_target Ascend --save_dir $SAVE_DIR