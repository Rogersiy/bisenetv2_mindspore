import os
import cv2
import multiprocessing
from mindspore import dataset as ds
from .cityscapes import citycapes_dataset
from .transforms_factory import create_transform


def create_dataset(cfg, num_parallel_workers=8, group_size=1, rank=0, task="train"):
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
    if cfg.name == "cityscapes":
        dataset = citycapes_dataset(dataset_dir=cfg.dataset_dir,
                                    map_label=cfg.map_label,
                                    ignore_label=cfg.ignore_label,
                                    group_size=group_size,
                                    rank=rank,
                                    is_train=is_train)
    else:
        NotImplementedError

    dataset = dataset.map(operations=transforms_list, input_columns=["image", "label"],
                          output_columns=["image", "label"],
                          python_multiprocessing=True)
    dataset = dataset.batch(cfg.batch_size, drop_remainder=is_train)
    return dataset, dataset.get_dataset_size()
