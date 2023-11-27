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

if [ $# != 2 ]
then
    echo "Using: bash scripts/run_eval.sh [DEVICE_ID] [CHECKPOINT_PATH]"
    exit 1
fi

get_real_path(){
    if [ "${1:0:1}" == "/" ]; then
        echo "$1"
    else
        echo "$(realpath -m $PWD/$1)"
    fi
}

PATH2=$(get_real_path $2)


if [ ! -f $PATH2 ]
then
    echo "error: CHECKPOINT_PATH=$PATH2 is not a file."
    exit 1
fi

export DEVICE_NUM=1
export DEVICE_ID=$1
export RANK_SIZE=$DEVICE_NUM
export RANK_ID=0

EXECUTE_PATH=$(pwd)
eval_path=${EXECUTE_PATH}/eval


if [ -d ${eval_path} ]; then
  rm -rf ${eval_path}
fi
mkdir -p ${eval_path}


env >${eval_path}/env.log
echo "start evaluation for device $DEVICE_ID"
python3 eval.py --checkpoint_path $PATH2 --save_dir ${eval_path} &> ${eval_path}/log &

