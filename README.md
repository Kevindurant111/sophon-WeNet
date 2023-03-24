# WeNet

## 目录

* [1. 简介](#1-简介)
* [2. 特性](#2-特性)
* [3. 准备模型与数据](#3-准备模型与数据)
* [4. 模型编译](#4-模型编译)
  * [4.1 TPU-NNTC编译BModel](#41-tpu-nntc编译bmodel)
* [5. 例程测试](#5-例程测试)
* [6. 精度测试](#6-精度测试)
  * [6.1 测试方法](#61-测试方法)
  * [6.2 测试结果](#62-测试结果)
* [7. 性能测试](#7-性能测试)
  * [7.1 bmrt_test](#71-bmrt_test)
  * [7.2 程序运行性能](#72-程序运行性能)
* [8. FAQ](#8-faq)

## 1. 简介
WeNet是一款面向工业落地应用的语音识别工具包，提供了从语音识别模型的训练到部署的一条龙服务。本例程对[WeNet官方开源仓库](https://github.com/wenet-e2e/wenet)中基于aishell的预训练模型和算法进行移植，使之能在SOPHON BM1684和BM1684X上进行推理测试。

## 2. 特性
* 支持BM1684X(x86 PCIe、SoC)和BM1684(x86 PCIe、SoC、arm PCIe)
* 支持FP32模型编译和推理
* 支持基于torchaudio的Python推理
* 支持单batch模型推理
* 支持流式语音的测试

## 3. 准备模型与数据
请使用TPU-NNTC编译BModel，Pytorch模型在编译前要导出成torchscript模型或onnx模型，本例程中提供了转换好的onnx模型。(TPU-MLIR的编译暂时不支持)

​同时，您需要准备用于测试的数据集。

​本例程在`scripts`目录下提供了相关模型和数据的下载脚本`download.sh`，您也可以自己准备模型和数据集，并参考[4. 模型编译](#4-模型编译)进行模型转换。

```bash
# 安装unzip，若已安装请跳过
sudo apt install unzip
chmod -R +x scripts/
./scripts/download.sh
```

下载的模型包括：
```
./models
├── BM1684
│   └── wenet_encoder_fp32.bmodel             # 使用TPU-NNTC编译，用于BM1684的FP32 BModel，batch_size=1
├── BM1684X
│   └── wenet_encoder_fp32.bmodel             # 使用TPU-NNTC编译，用于BM1684X的FP32 BModel，batch_size=1
└── onnx
    └── wenet_encoder.onnx                    # 导出的onnx模型       
```

下载的数据包括：
```
./datasets
├── aishell_S0764                                      
    ├── *.wav                                 # 从aishell数据集中抽取的用于测试的音频文件
    ├── aishell_S0764.list                    # 数据集的描述文件
    └── ground_truth.txt                      # 数据集标签文件，用于计算精度评价指标  
```

下载的Python拓展模块包括：
```
./swig_decoders
├── EGG-INFO                                       # 包含模块信息的文件夹
├── _swig_decoders.py                              # Python库文件   
├── swig_decoders.py                               # Python库文件
└── _swig_decoders.cpython-38-${arch}-linux-gnu.so # Python库依赖的动态链接库文件，arch表示机器架构，pcie和soc模式对应的arch是不同的                
```

## 4. 模型编译
导出的模型需要编译成BModel才能在SOPHON TPU上运行，如果使用下载好的BModel可跳过本节。

### 4.1 TPU-NNTC编译BModel
模型编译前需要安装TPU-NNTC，具体可参考[TPU-NNTC环境搭建](../../docs/Environment_Install_Guide.md#1-tpu-nntc环境搭建)。请注意，本例程编译模型使用的SDK需为最新的Release版本，安装好后需在TPU-NNTC环境中进入例程目录。

- 生成FP32 BModel

使用TPU-NNTC将onnx模型编译为FP32 BModel，具体方法可参考《TPU-NNTC开发参考手册》的“BMNETO 使用”(请从[算能官网](https://developer.sophgo.com/site/index/material/28/all.html)相应版本的SDK中获取)。

​本例程在`scripts`目录下提供了TPU-NNTC编译FP32 BModel的脚本，请注意修改`gen_fp32bmodel_nntc.sh`中的onnx模型路径、生成模型目录和输入大小shapes等参数，在执行时输入BModel的目标平台，如：

```bash
./scripts/gen_fp32bmodel_nntc.sh BM1684
```

​执行上述命令会在`models/BM1684/`下生成`wenet_encoder_fp32.bmodel`文件，即转换好的FP32 BModel。

## 5. 例程测试
- [Python例程](./python/README.md)

## 6. 精度测试
### 6.1 测试方法
首先，参考[Python例程](python/README.md#22-测试图片)推理要测试的数据集，生成预测的txt文件，注意修改数据集(datasets/aishell_S0764)和相关参数。  
然后，使用`tools`目录下的`eval_aishell.py`脚本，将测试生成的txt文件与测试集标签txt文件进行对比，计算出语音识别的评价指标，命令如下：
```bash
# 请根据实际情况修改程序路径和txt文件路径
python3 tools/eval_aishell.py --char=1 --v=1 datasets/aishell_S0764/ground_truth.txt python/result.txt  > online_wer
cat online_wer | grep "Overall"
```

### 6.2 测试结果
在coco2017val数据集上，精度测试结果如下：
|   测试平台    |    测试程序   |              测试模型     | WER    |
| ------------ | ------------ | ------------------------- | ------ |
| BM1684 PCIe  | wenet.py     | wenet_encoder_fp32.bmodel | 2.62%  |
| BM1684 SoC   | wenet.py     | wenet_encoder_fp32.bmodel | 2.62%  |
| BM1684X PCIe | wenet.py     | wenet_encoder_fp32.bmodel | 2.62%  | 
| BM1684X SoC  | wenet.py     | wenet_encoder_fp32.bmodel | 2.62%  | 

> **测试说明**：  
1. wer在不同的测试平台上是相同的。

## 7. 性能测试
### 7.1 bmrt_test
使用bmrt_test测试模型的理论性能：
```bash
# 请根据实际情况修改要测试的bmodel路径
bmrt_test --bmodel models/BM1684/wenet_encoder_fp32.bmodel
```
测试结果中的`calculate time`就是模型推理的时间。
测试各个模型的理论推理时间，结果如下：

|                  测试模型                   | calculate time(ms) |
| ------------------------------------------- | ----------------- |
| BM1684/wenet_encoder_fp32.bmodel            | 36.01             |
| BM1684X/wenet_encoder_fp32.bmodel           | 18.38             |

> **测试说明**：  
1. 性能测试结果具有一定的波动性；
2. `calculate time`已折算为1秒音频的推理时间。

### 7.2 程序运行性能
参考[C++例程](cpp/README.md)或[Python例程](python/README.md)运行程序，并查看统计的解码时间、预处理时间、推理时间、后处理时间。C++例程打印的预处理时间、推理时间、后处理时间为整个batch处理的时间，需除以相应的batch size才是每张图片的处理时间。

在不同的测试平台上，使用不同的例程、模型测试`datasets/test`，性能测试结果如下：
|    测试平台  |     测试程序      |             测试模型                |preprocess_time|inference_time|postprocess_time| 
| ----------- | ---------------- | ----------------------------------- | ------------- | --------- | ----------------- |
| BM1684 PCIe | wenet.py         | wenet_encoder_fp32.bmodel           | xxxx          | xxxx      | xxxx              |
| BM1684 SoC  | wenet.py         | wenet_encoder_fp32.bmodel           | 0.0014        | 47.12     | 10.00             |
| BM1684X PCIe| wenet.py         | wenet_encoder_fp32.bmodel           | 0.0002        | 27.61     | 3.25              |
| BM1684X SoC | wenet.py         | wenet_encoder_fp32.bmodel           | 0.0014        | 27.42     | 10.24             |


> **测试说明**：  
1. 时间单位均为毫秒(ms)，统计的时间均为1秒音频处理的时间；
2. 性能测试结果具有一定的波动性，建议多次测试取平均值；
3. BM1684/1684X SoC的主控CPU均为8核 ARM A53 42320 DMIPS @2.3GHz，PCIe上的性能由于CPU的不同可能存在较大差异；

## 8. FAQ  
1. ImportError: xxxx/libstdc++.so.6: version `GLIBCXX_3.4.30' not found: 常出现在pcie模式下，原因是编译好的ctc decoder与本机的环境不适配。  
解决方法：在要运行的本机上重新编译一份ctc decoder。
```bash
git clone https://github.com/Slyne/ctc_decoder.git  
apt-get update
apt-get install swig
apt-get install python3-dev 
cd ctc_decoder/swig && bash setup.sh
```