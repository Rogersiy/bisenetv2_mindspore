from mindspore import dataset as ds


def citycapes_dataset(dataset_dir, map_label=True, ignore_label=255, group_size=1, rank=0, is_train=True):
    usage = "train" if is_train else "val"
    dataset = ds.CityscapesDataset(dataset_dir=dataset_dir, usage=usage, task="semantic", decode=True,
                                   num_shards=group_size, shard_id=rank, shuffle=is_train)
    if map_label:
        def convert_label(label, inverse=False):
            """Convert classification ids in labels."""
            temp = label.copy()
            if inverse:
                for v, k in label_mapping.items():
                    label[temp == k] = v
            else:
                for k, v in label_mapping.items():
                    label[temp == k] = v
            return label

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
        dataset = dataset.map(operations=convert_label, input_columns=["task"], output_columns=["label"])
    return dataset
