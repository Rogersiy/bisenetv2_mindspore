import os
import argparse
import ast
import traceback
import numpy as np
import mindspore as ms
from mindspore import nn

from src.modules.ocrnet import OCRNet
from src.modules.loss import WithLossCell, CrossEntropy
from src.data.dataset_factory import create_dataset
from src.utils import logger
from src.utils.config import load_config, Config, merge
from src.utils.callback import Callback
from src.utils.common import init_env, clear
from src.modules.train_warpper import TrainOneStepCell


def get_args_train(parents=None):
    parser = argparse.ArgumentParser(description='Train', parents=[parents] if parents else [])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config", type=str,
                        default=os.path.join(current_dir,"config/config_ocrnet_hrw48.yml"),
                        help="Config file path")
    parser.add_argument('--seed', type=int, default=1234, help='runtime seed')
    parser.add_argument('--ms_mode', type=int, default=0,
                        help='Running in GRAPH_MODE(0) or PYNATIVE_MODE(1) (default=0)')
    parser.add_argument('--ms_loss_scaler', type=str, default='static', help='train loss scaler, static/dynamic/none')
    parser.add_argument('--ms_loss_scaler_value', type=float, default=256.0, help='static loss scale value')
    parser.add_argument('--device_target', type=str, default='Ascend', help='device target, Ascend/GPU/CPU')
    parser.add_argument('--pre_trained_ckpt', type=str, default='', help='pre trained weights path')
    parser.add_argument('--batch_size', type=int, default=2, help='total batch size for all device')
    parser.add_argument('--mix', type=ast.literal_eval, default=True, help='Mix Precision')
    parser.add_argument('--run_eval', type=ast.literal_eval, default=True, help='run eval')
    parser.add_argument('--clip_grad', type=ast.literal_eval, default=False, help='clip grad')
    parser.add_argument('--force_update', type=ast.literal_eval, default=False, help='force update')
    parser.add_argument('--eval_parallel', type=ast.literal_eval, default=True, help='run eval')
    parser.add_argument('--log_interval', type=int, default=100, help='log interval')
    parser.add_argument('--lr_init', type=float, default=0.02, help='base learning rate')
    parser.add_argument('--save_dir', type=str, default="output", help='save dir')

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


def get_lr(lr_init, end_lr, warmup_step, total_step):
    assert warmup_step < total_step
    w_r = (lr_init - end_lr) / warmup_step
    d_r = (lr_init - end_lr) / (total_step - warmup_step)
    lrs = []
    for i in range(total_step):
        if i < warmup_step:
            lrs.append(end_lr + i * w_r)
        else:
            lrs.append(lr_init - ((i - warmup_step) * d_r))
    return np.array(lrs, np.float32)


def get_optimizer(cfg, params, lr):
    def init_group_params(params, weight_decay):
        decay_params = []
        no_decay_params = []

        for param in params:
            if "beta" not in param.name and "gamma" not in param.name and "bias" not in param.name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
        return [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params},
            {"order_params": params},
        ]
    weight_decay = cfg.weight_decay
    if cfg.filter_bias_and_bn:
        params = init_group_params(params, weight_decay)
        weight_decay = 0.0

    if cfg.type in ["momentum", "sgd"]:
        opt = nn.Momentum(params, lr, momentum=cfg.momentum, weight_decay=weight_decay, nesterov=cfg.nesterov)
        return opt
    raise ValueError(f"Not support {cfg.type}")


def train(cfg, network, dataset, eval_dataset=None):

    model = ms.Model(train_net)
    epochs = cfg.total_step // cfg.log_interval
    model.train(epochs, dataset, callbacks=Callback(cfg, network, optimizer, eval_dataset),
                dataset_sink_mode=True, sink_size=cfg.log_interval)
    if cfg.enable_modelarts:
        from src.utils.modelarts import sync_data
        sync_data(cfg.save_dir, cfg.train_url)


if __name__ == '__main__':
    args = get_args_train()
    config, helper, choices = load_config(args.config)
    config = merge(args, config)
    config = Config(config)
    init_env(config)

    # Dataset
    dataloader, steps_per_epoch = create_dataset(config.data,
                                                 num_parallel_workers=config.data.num_parallel_workers,
                                                 group_size=config.rank_size,
                                                 rank=config.rank,
                                                 task="train")
    config.steps_per_epoch = steps_per_epoch
    eval_dataloader = None

    # Network
    network = OCRNet(config)
    network.to_float(ms.float32)
    if config.mix:
        network.backbone.to_float(ms.float16)
        network.fcn_head.to_float(ms.float16)
        for _, cell in network.cells_and_names():
            if isinstance(cell, (nn.Dense)):
                cell.to_float(ms.float16)
    loss_fn = CrossEntropy(num_classes=config.num_classes,
                           ignore_label=config.data.ignore_label,
                           cls_weight=config.data.map_label).to_float(ms.float32)
    net_with_loss = WithLossCell(network, loss_fn)

    # Optimizer
    total_step = (config.total_step // steps_per_epoch + 1) * steps_per_epoch
    lr = get_lr(config.lr_init, 1e-6, config.warmup_step, total_step)
    optimizer = get_optimizer(config.optimizer, net_with_loss.trainable_params(), lr)
    scale_sense = nn.FixedLossScaleUpdateCell(1.0)
    if config.ms_loss_scaler == 'dynamic':
        scale_sense = nn.DynamicLossScaleUpdateCell(loss_scale_value=config.get('ms_loss_scaler_value', 2 ** 16),
                                                    scale_factor=config.get('scale_factor', 2),
                                                    scale_window=config.get('scale_window', 2000))
    elif config.ms_loss_scaler == 'static':
        scale_sense = nn.FixedLossScaleUpdateCell(config.get('ms_loss_scaler_value', 2 ** 10))
    train_net = TrainOneStepCell(network, optimizer, scale_sense,
                                 clip_grad=config.clip_grad, force_update=config.force_update)

    if os.path.exists(config.pre_trained_ckpt):
        ms.load_checkpoint(config.pre_trained_ckpt, train_net)
        logger.info(f"success to load pretrained ckpt {config.pre_trained_ckpt}")


    # if config.run_eval:
    #     eval_dataloader, _ = create_dataloader(data_config=config.data,
    #                                            task='eval',
    #                                            per_batch_size=1,
    #                                            rank=config.rank, rank_size=config.rank_size,
    #                                            shuffle=False, drop_remainder=False,
    #                                            num_parallel_worker=1)

    try:
        train(config, network, dataloader, eval_dataloader)
    except:
        traceback.print_exc()
    finally:
        clear(enable_modelarts=config.enable_modelarts,
              save_dir=config.save_dir,
              train_url=config.train_url)
