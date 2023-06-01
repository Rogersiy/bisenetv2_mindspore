import time
import numpy as np
import mindspore as ms
from src.modules.ocrnet import OCRNet
from src.utils.config import load_config, Config, merge
from src.data import create_dataset


if __name__ == '__main__':
    config, helper, choices = load_config("config/ocrnet/config_ocrnet_hrw48.yml")
    config = Config(config)
    net = OCRNet(config)

    img = ms.Tensor(np.random.uniform(-1, 1, (2, 3, 1280, 768)), ms.float32)
    fcn_out, ocr_out = net(img)
    print(fcn_out.shape, ocr_out.shape)
    config.batch_size = 2
    dataloader, steps_per_epoch = create_dataset(config.data,
                                                 num_parallel_workers=config.data.num_parallel_workers,
                                                 group_size=1,
                                                 rank=0,
                                                 task="train")
    print("steps_per_epoch", steps_per_epoch)
    s = time.time()
    for i, data in enumerate(dataloader.create_dict_iterator(output_numpy=True)):
        if i == 0:
            print(data["image"].shape, data["label"].shape)
        elif i == 5:
            s = time.time()
        elif i == 55:
            print("avg time", (time.time() - s) / 50 * 1000)
