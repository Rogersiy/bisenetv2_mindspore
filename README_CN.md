# 目录

<!-- TOC -->

- [目录](#目录)
- [BiSeNetv2描述](#BiSeNetv2描述)
  - [概述](#概述)
  - [论文](#论文)
- [模型架构](#模型架构)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
  - [脚本及样例代码](#脚本及样例代码)
  - [脚本参数](#脚本参数)
  - [训练过程](#训练过程)
    - [用法](#用法)
      - [Ascend处理器环境运行](#ascend处理器环境运行)
      - [训练时推理](#训练时推理)
    - [结果](#结果)
  - [评估过程](#评估过程)
    - [用法](#用法-1)
      - [Ascend处理器环境运行](#ascend处理器环境运行-1)
    - [结果](#结果-1)
  - [推理过程](#推理过程)
    - [导出MindIR](#导出mindir)
    - [执行推理](#执行推理)
    - [结果](#结果-2)
- [模型描述](#模型描述)
  - [性能](#性能)
    - [评估性能](#评估性能)
      - [Cityscapes上OCRNet的性能](#cityscapes上ocrnet的性能)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)
- [FAQ](#FAQ)

<!-- /TOC -->

# BiSeNetv2描述

## 概述

BiSeNetv2是一个轻量级的双边语义分割网络，一条路径被设计为通过宽通道和浅层次来捕获空间细节，被称为细节分支（Detail Branch）。相反，另一条路径被引入来通过窄通道和深层次提取类别语义，被称为语义分支（Semantic Branch）。语义分支只需要一个大的感受野来捕获语义上下文，而细节信息可以由细节分支提供。因此，语义分支可以通过较少的通道和快速下采样策略制作得非常轻量级。两种类型的特征表示被合并以构造一个更强大、更全面的特征表示。

## 论文

[BiSeNet V2: Bilateral Network with Guided Aggregation for Real-time Semantic Segmentation](https://arxiv.org/pdf/2004.02147.pdf)

# 模型架构

BiSeNetv2的总体架构如下:

![BiSeNetv2](BiSeNetv2.png)

# 数据集

1. 数据集[Cityscapes](https://www.cityscapes-dataset.com/)

Cityscapes数据集包含5000幅高质量像素级别精细注释的街城市道场景图像。图像按2975/500/1525的分割方式分为三组，分别用于训练、验证和测试。数据集中共包含30类实体，其中19类用于验证。

2. 数据集下载后的结构模式

```bash
$SEG_ROOT/data
├─ cityscapes
│  ├─ leftImg8bit
│  │  ├─ train
│  │  │  └─ [city_folders]
│  │  └─ val
│  │     └─ [city_folders]
│  ├─ gtFine
│  │  ├─ train
│  │  │  └─ [city_folders]
│  │  └─ val
│  │     └─ [city_folders]
│  ├─ train.lst
│  └─ val.lst

```

# 环境要求

- 硬件（Ascend）
  - 准备Ascend处理器搭建硬件环境
- Mindspore版本依赖 2.0alpha
  - [Mindspore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
  - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
  - [MindSpore Python API](https://www.mindspore.cn/docs/zh-CN/master/index.html)

# 快速入门

修改 `config/cityscapes.yml` 中 Cityscapes 数据集的路径：

```
dataset_dir: "your cityscapes"
```



- Ascend处理器环境运行

```bash
# 分布式训练 4卡环境 batchsize设置为4 ，8卡环境 batchsize设置为2
bash scripts/run_distribute_train.sh [DEVICE_NUM] [BATCH_SIZE] [SAVE_DIR]

# 单机训练
bash scripts/run_standalone_train.sh [SAVE_DIR]

# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [CHECKPOINT_PATH]
```

如果要在ModelArts上进行模型的训练，可以参考ModelArts的[官方指导文档](https://support.huaweicloud.com/modelarts/)开始进行模型的训练，具体操作如下：

```text
# 训练模型 
1. 创建作业
2. 选择数据集存储位置
3. 选择输出存储位置
2. 在模型参数列表位置按如下形式添加参数：
    data_url            [自动填充]
    train_url           [自动填充]
    checkpoint_url      [CHECKPOINT_PATH_OBS]
    modelarts           True
    device_target       Ascend
    #batchsize根据计算节点数设置，建议总batchsize为16
    # 其他可选参数具体详情请参考train.py脚本
3. 选择相应数量的处理器
4. 开始运行
```

# 脚本说明

## 脚本及样例代码

```bash
├─ BiSeNetv2
│  ├─ ascend310_infer                       # 310推理相关脚本
│  │  ├─ inc
│  │  │  └─ utils.py
│  │  └─ src
│  │  │  ├─ build.sh
│  │  │  ├─ CMakeLists.txt
│  │  │  ├─ main.cc
│  │  │  └─ utils.cc
│  ├─ scripts
│  │  ├─ ascend_310_infer.sh                # 启动Ascend310推理（单卡）
│  │  ├─ run_standalone_train.sh            # 启动Ascend单机训练（单卡）
│  │  ├─ run_distribute_train.sh            # 启动Ascend分布式训练（多卡）
│  │  └─ run_eval.sh                        # 启动Asend910单机评估（单卡）
├── src
│   ├── data
│   │   ├── cityscapes.py					# Cityscapes数据定义
│   │   ├── dataset_factory.py				# 创建Cityscapes数据集
│   │   ├── __init__.py
│   │   ├── transforms
│   │   │   ├── common.py        			#数据集预处理操作定义
│   │   │   ├── __init__.py
│   │   │   └── resize.py					#数据集resize操作定义
│   │   └── transforms_factory.py
│   ├── __init__.py
│   ├── modules
│   │   ├── base_modules.py					#多尺度推理类
│   │   ├── bisenetv2.py					#BiSeNetv2网络结构
│   │   ├── __init__.py
│   │   ├── loss
│   │   │   ├── cross_entropy.py  			#交叉熵损失函数定义
│   │   │   ├── __init__.py
│   │   │   └── loss.py						#WithLossCell
│   │   └── train_warpper.py
│   └── utils
│       ├── callback.py						#训练时回调函数
│       ├── common.py						#环境初始化
│       ├── config.py						#配置文件处理
│       ├── __init__.py
│       ├── local_adapter.py				#本地适配
│       ├── logger.py						#日志输出
│       ├── metrics.py						#评估计算
│       └── modelarts.py					#适配modelarts
│  ├─ export.py                             # 310推理，导出mindir
│  ├─ preprocess.py                         # 310推理，数据预处理
│  ├─ postprocess.py                        # 310推理，计算mIoU
│  ├─ train.py                              # 训练模型
│  └─ eval.py                               # 评估模型
```

## 脚本参数

在配置文件中可以同时配置训练参数和评估参数。

```python
net: BiSeNetV2											#模型名称		
num_classes: 19											#类别数量
loss_weight: [1.0, 1.0,1.0,1.0,1.0]						#损失函数权重
# ===== Dataset ===== #
data:
  dataset_dir: "Cityscapes"								#数据集存放位置
  name: "cityscapes"									#数据集名称
  num_parallel_workers: 8								
  map_label: [0.8373, 0.918, 0.866, 1.0345,
              1.0166, 0.9969, 0.9754, 1.0489,
              0.8786, 1.0023, 0.9539, 0.9843,
              1.1116, 0.9037, 1.0865, 1.0955,
              1.0865, 1.1529, 1.0507]
  ignore_label: 255										#不被考虑的类别标签值
  train_transforms:
    item_transforms:
      - RandomResizeCrop: { 							#随机resize裁剪
          crop_size: [1024, 1024], 
          multi_scale: True, 
          base_size: 2048, 
          ignore_label: 255}
      - RandomFlip: { prob: 0.5 }						#数据增强参数
      - RandomColor: { contrast_range: [0.4, 1.6] }		#数据增强参数
      - Normalize: { is_scale: True, 					#数据增强参数
                    norm_type: 'mean_std',
                    mean: [0.3257, 0.3690, 0.3223], 
                    std: [0.2112, 0.2148, 0.2115] }
      - TransposeImage: { hwc2chw: True }				#hwc转为chw

  eval_transforms:
    multi_scale: True									#是否进行多尺度推理
    img_ratios: [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]		#多尺度
    flip: True											#是否翻转
    item_transforms:
      - Resize: { target_size: [1024, 2048], 			#数据增强参数
                 keep_ratio: True, 
                 ignore_label: 255 }
      - Normalize: { is_scale: True, 					#数据增强参数
                    norm_type: 'mean_std',  
                    mean: [0.3257, 0.3690, 0.3223], 
                    std: [0.2112, 0.2148, 0.2115] }
      - TransposeImage: { hwc2chw: True }				#hwc转为chw
# ===== Learning Rate Policy ======== #
optimizer:
  type: sgd												#优化器使用sgd
  momentum: 0.9											#优化器动量
  weight_decay: 0.0005									#下降decay系数
  nesterov: False										#是否使用牛顿法收敛
  filter_bias_and_bn: False								#是否过滤bias和bn
warmup_step: 1000										#预热迭代数
total_step: 160000										#总迭代数

```

## 训练过程

### 用法

#### Ascend处理器环境运行

```bash
# 分布式训练 4卡环境 batchsize设置为4 ，8卡环境 batchsize设置为2
bash scripts/run_distribute_train.sh [DEVICE_NUM] [BATCH_SIZE] [SAVE_DIR]

# 单机训练
bash scripts/run_standalone_train.sh [SAVE_DIR]

```

```bash
#python脚本运行训练
#单卡训练
python train.py 

#多卡训练
mpirun --allow-run-as-root -n [DEVICE_NUM] --merge-stderr-to-stdout \
    python train.py --config [CONFIG_PATH] --device_target [DEVICE_TARGET] 
    

```



#### 训练时推理

如果需要训练时推理，在执行shell脚本前，修改shell脚本或以python脚本方式启动训练，给train.py 添加--run_eval True即可

### 结果

使用Cityscapes数据集训练BiSeNetv2

```text
# 分布式训练结果（4p）
 Start Train
2023-07-10 23:52:42,198 [INFO] Start Training
2023-07-10 23:56:49,211 [INFO] step 10/160000, loss: 14.1842, cur_lr: 0.049997, cost 24701.19 ms
2023-07-10 23:56:56,564 [INFO] step 20/160000, loss: 8.2247, cur_lr: 0.049995, cost 735.22 ms
2023-07-10 23:57:03,880 [INFO] step 30/160000, loss: 7.8817, cur_lr: 0.049992, cost 731.52 ms
2023-07-10 23:57:11,176 [INFO] step 40/160000, loss: 10.0769, cur_lr: 0.049989, cost 729.53 ms
2023-07-10 23:57:18,506 [INFO] step 50/160000, loss: 5.7104, cur_lr: 0.049986, cost 732.93 ms
2023-07-10 23:57:25,802 [INFO] step 60/160000, loss: 6.0667, cur_lr: 0.049983, cost 729.64 ms
2023-07-10 23:57:33,108 [INFO] step 70/160000, loss: 7.1434, cur_lr: 0.049981, cost 730.55 ms
...
```

## 评估过程

### 用法

#### Ascend处理器环境运行

```bash
# 运行评估
bash scripts/run_eval.sh [DEVICE_ID] [CHECKPOINT_PATH]

#python运行 
python eval.py 
#更多参数输入见eval.py
```

### 结果

评估结果保存在示例路径中，文件夹名为“eval”。你可在此路径下的日志文件中找到如下结果：

```text
2023-08-03 11:38:25,144 [INFO] Total number of images: 500
2023-08-03 11:38:25,146 [INFO] =========== Evaluation Result ===========
2023-08-03 11:38:25,148 [INFO] iou array: 
 [0.97665877 0.81983653 0.91559855 0.53803037 0.55758281 0.60849648
 0.66469014 0.75209206 0.91862638 0.61609251 0.94315898 0.78908622
 0.57386983 0.93849353 0.57065237 0.7203408  0.70186687 0.54107484
 0.74231258]
2023-08-03 11:38:25,148 [INFO] miou: 0.7309768747454519
```

## 推理过程

**推理前需参照 [MindSpore C++推理部署指南](https://gitee.com/mindspore/models/blob/master/utils/cpp_infer/README_CN.md) 进行环境变量设置。**

### 导出MindIR

```bash
python export.py --device_id [DEVICE_ID] --device_target Ascend --checkpoint_file [CKPT_PATH] --file_name [FILE_NAME] --file_format MINDIR 
```

### 执行推理

在执行推理之前，必须先通过`export.py`脚本到本mindir文件。以下展示了使用mindir模型执行推理的示例。目前只支持Cityscapes数据集batchsize为1的推理。

```bash
bash scripts/ascend310_inference.sh [MINDIR_PATH] [DATA_PATH] [DEVICE_ID]
```

- `MINDIR_PATH` mindir文件的存储路径
- `DATA_PATH` Cityscapes原始数据集的存储路径
- `DEVICE_TYPE` 可以为Ascend, GPU, 或CPU。
- `DEVICE_ID` 卡号

脚本内部分为三步：

1. `preprocess.py`对原始数据集进行预处理，并将处理后的数据集以二进制的形式存储在`./preprocess_Result/`路径下；
2. `ascend310_infer/src/main.cc`执行推理过程，并将预测结果以二进制的形式存储在`./result_Files/`路径下，推理日志可在`infer.log`中查看；
3. `postprocess.py`利用预测结果与相应标签计算mIoU，计算结果可在`acc.log`中查看。

### 结果

```text
Total number of images:  500
=========== 310 Inference Result ===========
miou: 0.6936929179415765
iou array: 
 [0.97183646 0.7859067  0.89604381 0.52313834 0.54708609 0.44391835
 0.58659314 0.6738844  0.89592125 0.59430689 0.91434377 0.72914479
 0.53376993 0.9126913  0.56362668 0.70687338 0.69249336 0.51109139
 0.69749539]
============================================
```

# 模型描述

## 性能

### 评估性能

#### Cityscapes上BiSeNetv2的性能

| 参数          | Ascend 910                                               |
| ------------- | -------------------------------------------------------- |
| 模型版本      | BiSeNetv2                                                |
| 资源          | Ascend 910；CPU 2.60GHz，192核；内存 755G；系统 Euler2.8 |
| 上传日期      | 2023/8/3                                                 |
| MindSpore版本 | 2.0alpha                                                 |
| 数据集        | Cityscapes                                               |
| 训练参数      | total_step=160000 , batch_size = 4                       |
| 优化器        | SGD                                                      |
| 损失函数      | CE交叉熵损失函数                                         |
| 输出          | mIoU                                                     |
| 损失          | 0.06756218                                               |
| 速度          | 728毫秒/步（4卡）                                        |
| 总时长        | 33h                                                      |

# 随机情况说明

`train.py`中使用了随机种子。

# 免责说明

models仅提供转换模型的脚本。我们不拥有这些模型，也不对它们的质量负责和维护。对这些模型进行转换仅用于非商业研究和教学目的。

致模型拥有者：如果您不希望将模型包含在MindSpore models中，或者希望以任何方式对其进行转换，我们将根据要求删除或更新所有公共内容。请通过Gitee与我们联系。非常感谢您对这个社区的理解和贡献。

# ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。

# FAQ

Q：Mindspore使用2.0或以上版本，网络架构中interpolate报错，

A：2.0alpha版本interpolate参数定义是sizes，2.0开始变成size