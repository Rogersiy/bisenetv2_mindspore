import os
import cv2
import copy
import multiprocessing
from mindspore import dataset as ds
from .cityscapes import citycapes_dataset
from .transforms_factory import create_transform
from .transforms import Resize, RandomFlip


def create_dataset(cfg, batch_size, num_parallel_workers=8, group_size=1, rank=0, task="train"):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    cv2.setNumThreads(2)
    ds.config.set_enable_shared_mem(True)
    cores = multiprocessing.cpu_count()
    num_parallel_workers = min(num_parallel_workers, cores // group_size)
    ds.config.set_num_parallel_workers(num_parallel_workers)
    is_train = task == 'train'
    if task == 'train':
        trans_config = getattr(cfg, 'train_transforms', cfg)
    elif task in ('val', 'eval'):
        trans_config = getattr(cfg, 'eval_transforms', cfg)
    else:
        raise NotImplementedError
    item_transforms = getattr(trans_config, 'item_transforms', [])
    transforms_name_list = []
    for transform in item_transforms:
        transforms_name_list.extend(transform.keys())
    transforms_list = []
    for i, transform_name in enumerate(transforms_name_list):
        transform = create_transform(item_transforms[i])
        transforms_list.append(transform)
    ori_dataset = None
    if cfg.name == "cityscapes":
        ori_dataset = citycapes_dataset(dataset_dir=cfg.dataset_dir,
                                        map_label=cfg.map_label,
                                        ignore_label=cfg.ignore_label,
                                        group_size=group_size,
                                        rank=rank,
                                        is_train=is_train)
    else:
        NotImplementedError
    if task == 'train':
        dataset = ori_dataset.map(operations=transforms_list, input_columns=["image", "label"],
                                  python_multiprocessing=True)
    else:
        datasets = []
        base_size = trans_config.base_size
        if trans_config.multi_scale or trans_config.random_flip:
            random_lists = []
            for r in trans_config.img_ratios:
                target_size = [int(base_size[0] * r), int(base_size[1] * r)]
                resize = Resize(target_size=target_size,
                                keep_ratio=True,
                                ignore_label=cfg.ignore_label)
                random_lists.append([resize])
                if trans_config.random_flip:
                    flip = RandomFlip(1.0)
                    random_lists.append([resize, flip])
            for random_list in random_lists:
                item = copy.deepcopy(ori_dataset).map(operations=random_list + transforms_list,
                                                      input_columns=["image", "label"],
                                                      python_multiprocessing=True)
                item = item.project(["image", "label", "ori_shape"])
                item = item.batch(batch_size, drop_remainder=False)
                datasets.append(item)
        else:
            resize = Resize(target_size=base_size,
                            keep_ratio=True,
                            ignore_label=cfg.ignore_label)
            item = ori_dataset.map(operations=[resize] + transforms_list,
                                   input_columns=["image", "label"],
                                   python_multiprocessing=True)
            item = item.project(["image", "label", "ori_shape"])
            item = item.batch(batch_size, drop_remainder=False)
            datasets.append(item)
        return datasets, len(datasets)
    dataset = dataset.project(["image", "label"])
    dataset = dataset.batch(batch_size, drop_remainder=is_train)
    return dataset, dataset.get_dataset_size()
