import os
import argparse
import ast
import traceback
import numpy as np
import mindspore as ms

from src.modules.base_modules import MultiScaleInfer
from src.data.dataset_factory import create_dataset
from src.utils import logger
from src.utils.config import load_config, Config, merge
from src.utils.common import init_env, clear
from src.utils.metrics import get_confusion_matrix
from src.data.visualize import visualize


def get_args_train(parents=None):
    parser = argparse.ArgumentParser(description="Train", parents=[parents] if parents else [])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(current_dir, "config/bisenetv2/config_bisenetv2_16k.yml"),
        help="Config file path",
    )
    
    parser.add_argument("--bin_path", type=str,default='./bin', help="Storage path of bin files.")

    
    parser.add_argument("--seed", type=int, default=1234, help="runtime seed")
    parser.add_argument(
        "--ms_mode", type=int, default=0, help="Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)"
    )
    parser.add_argument("--visualize", type=ast.literal_eval, default=False, help="visualize when eval")
    parser.add_argument("--device_target", type=str, default="Ascend", help="device target, Ascend/GPU/CPU")
    parser.add_argument("--checkpoint_path", type=str, default="./openi/BiSeNetV2_80000_rank0.ckpt", help="pre trained weights path")
    parser.add_argument("--eval_parallel", type=ast.literal_eval, default=True, help="run eval")
    parser.add_argument("--save_dir", type=str, default="output", help="save dir")
    parser.add_argument("--mix", type=ast.literal_eval, default=True, help="Mix Precision")

    # profiling
    parser.add_argument("--run_profilor", type=ast.literal_eval, default=False, help="run profilor")

    # args for ModelArts
    parser.add_argument("--enable_modelarts", type=ast.literal_eval, default=False, help="enable modelarts")
    parser.add_argument("--data_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--ckpt_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--train_url", type=str, default="", help="ModelArts: obs path to dataset folder")
    parser.add_argument("--data_dir", type=str, default="/cache/data", help="ModelArts: obs path to dataset folder")
    args, _ = parser.parse_known_args()
    return args




if __name__ == "__main__":
    args = get_args_train()
    config, helper, choices = load_config(args.config)
    config = merge(args, config)
    config = Config(config)
    init_env(config)

    # Dataset
    dataset, num = create_dataset(
        config.data,
        batch_size=1,
        num_parallel_workers=config.data.num_parallel_workers,
        task="eval",
        group_size=config.rank_size,
        rank=config.rank,
    )
    
    image_path = os.path.join(args.bin_path, "image")
    label_path = os.path.join(args.bin_path, "label")
    os.makedirs(image_path)
    os.makedirs(label_path)
    
    data_loader = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)

    for i ,data in enumerate(data_loader):
        image = ms.Tensor(data["image"])
        label = data["label"]
        file_name = "cityscapes_val_" + str(i) + ".bin"
        image_file_path = os.path.join(image_path, file_name)
        label_file_path = os.path.join(label_path, file_name)
        image.tofile(image_file_path)
        label.tofile(label_file_path)
    print("Export bin files finished!", flush=True)

