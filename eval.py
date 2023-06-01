import os
import argparse
import ast
import traceback
import cv2
import numpy as np
import mindspore as ms
from mindspore import nn, ops

from src.modules.ocrnet import OCRNet
from src.data.dataset_factory import create_dataset
from src.utils import logger
from src.utils.config import load_config, Config, merge
from src.utils.common import init_env, clear
from src.utils.metrics import get_confusion_matrix


def get_args_train(parents=None):
    parser = argparse.ArgumentParser(description='Train', parents=[parents] if parents else [])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config", type=str,
                        default=os.path.join(current_dir, "config/ocrnet/config_ocrnet_hrw48.yml"),
                        help="Config file path")
    parser.add_argument('--seed', type=int, default=1234, help='runtime seed')
    parser.add_argument('--ms_mode', type=int, default=0,
                        help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--checkpoint_path', type=str, default='', help='pre trained weights path')
    parser.add_argument('--eval_parallel', type=ast.literal_eval, default=True, help='run eval')
    parser.add_argument('--save_dir', type=str, default="output", help='save dir')
    parser.add_argument('--mix', type=ast.literal_eval, default=True, help='Mix Precision')

    # profiling
    parser.add_argument('--run_profilor', type=ast.literal_eval, default=False, help='run profilor')

    # args for ModelArts
    parser.add_argument('--enable_modelarts', type=ast.literal_eval, default=False, help='enable modelarts')
    parser.add_argument('--data_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--ckpt_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--train_url', type=str, default='', help='ModelArts: obs path to dataset folder')
    parser.add_argument('--data_dir', type=str, default='/cache/data', help='ModelArts: obs path to dataset folder')
    args, _ = parser.parse_known_args()
    return args


def postprocess(label, pred, ori_shape):
    """
    Args:
        label (np.array): Shape is (h, w). type is int32
        pred (np.array): Shape is (c, h, w).
        ori_shape (np.array): ori image shape, h, w
    """
    h, w = label.shape
    ori_h, ori_w = ori_shape
    im_scale = 1 / min(w / ori_w, h / ori_h)
    label = label.astype(np.uint8)
    label_ori = cv2.resize(label, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_NEAREST)
    pred = np.transpose(pred, (1, 2, 0))
    pred = np.argmax(pred, axis=2).astype(np.uint8)
    s = label_ori.shape
    pred_ori = cv2.resize(pred, (s[1], s[0]), interpolation=cv2.INTER_NEAREST)
    return label_ori, pred_ori


def run_eval(cfg, net, eval_datasets):
    num_classes = cfg.num_classes

    for dataset in eval_datasets:
        confusion_matrix = np.zeros((num_classes, num_classes))
        item_count = 0
        data_loader = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
        h, w = 0, 0
        for data in data_loader:
            img = ms.Tensor(data["image"])
            label = data["label"]
            _, _ , h, w = img.shape
            pred = net(img)
            if isinstance(pred, tuple):
                pred = pred[0]
            label, pred = postprocess(np.squeeze(label),
                                      np.squeeze(pred.asnumpy()),
                                      np.squeeze(data["ori_shape"]))
            confusion_matrix += get_confusion_matrix(label, pred, num_classes, ignore=cfg.data.ignore_label)
            item_count += 1
        logger.info(f"Total number of images: {item_count}")

        pos = confusion_matrix.sum(1)
        res = confusion_matrix.sum(0)
        tp = np.diag(confusion_matrix)
        iou_array = (tp / np.maximum(1.0, pos + res - tp))
        mean_iou = iou_array.mean()

        # Show results
        logger.info(f"==========={h}, {w} Evaluation Result ===========")
        logger.info(f"iou array: \n {iou_array}")
        logger.info(f"miou: {mean_iou}")


if __name__ == '__main__':
    args = get_args_train()
    config, helper, choices = load_config(args.config)
    config = merge(args, config)
    config = Config(config)
    init_env(config)

    # Dataset
    datasets, num = create_dataset(config.data,
                                   batch_size=1,
                                   num_parallel_workers=config.data.num_parallel_workers,
                                   task="eval")

    # Network
    network = OCRNet(config).set_train(False)
    if config.mix:
        network.to_float(ms.float16)

    ms.load_checkpoint(config.checkpoint_path, network)
    logger.info(f"success to load pretrained ckpt {config.checkpoint_path}")
    try:
        run_eval(config, network, datasets)
    except:
        traceback.print_exc()
    finally:
        clear(enable_modelarts=config.enable_modelarts,
              save_dir=config.save_dir,
              train_url=config.train_url)
