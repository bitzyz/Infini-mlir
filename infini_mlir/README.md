# infini_mlir

## 项目简介

本项目属于InfiniTensor下的一个子项目，旨在为原项目提供一个新的前端，使其在功能上更加完善与更具可扩展性。本项目通过使用mlir编译器基础设施，对神经网络模型进行解析与优化，从而获得更好的性能。

## 项目设计

本项目的设计思路大致为：

1、利用MLIR的python接口将一个神经网络模型解析为mlir文件与权重文件。

2、对得到的中间表示进行优化，包括形状推导、类型推导、图优化等。

3、将优化后的中间表示导出为InfiniTensor内部的计算图表示，供后端使用。

各子目录介绍如下：

- CAPI：提供方言注册接口供python前端使用。
- Dialect：IR目录提供了Infini Dialect的定义，Interfaces目录提供了各算子拥有接口的具体实现，Canonicalize目录提供了各算子的规范化模式实现，Transforms目录提供了各优化Pass的定义。
- Interfaces：提供了各算子需要实现的接口定义。
- python：项目的前端脚本文件等。
- Support：提供了module、tensorfile等数据结构的定义，用以支持该项目。
- Tools：用于调试、开发、优化等的一些工具。
- Traits：提供了算子所拥有的trait定义。
- Translation：与主项目的接口，用于将优化的mlir中间表示导出为计算图。
- include && src：提供方言注册与Pass注册的函数。

## 使用说明

### 构建命令

- `make`/`make build`: 将子项目与主项目一起构建。

### 模型转换与优化

```
// 目前已验证resnet18-v2-7.onnx模型
cd infini_mlir/python/translation
// 将其下载到python/translation下
wget https://github.com/onnx/models/raw/refs/heads/main/validated/vision/classification/resnet/model/resnet18-v2-7.onnx
// 执行转换脚本，并进行优化
python model_translator.py --model_name resnet --model_file resnet18-v2-7.onnx --input_shapes "[[1, 3, 244, 244]]" --mlir output.mlir
// 将权重文件移动至构建目录
mv resnet_origin_weight.npz ../../../build/Release/
// 返回主目录并执行make test-cpp
cd ../../..
make test-cpp
```

