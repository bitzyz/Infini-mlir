#include "ConvertToInfini.h"
#include "utils.h"

namespace infini {

namespace infinimlir {
using mlir::IntegerAttr;

Graph convertMLIRToInfini(mlir::ModuleOp module, Runtime runtime) {
    llvm::DenseMap<mlir::Value, Tensor> tensorMap;
    Graph g = make_ref<GraphObj>(runtime);
    for (auto func : module.getOps<mlir::func::FuncOp>()) {
        for (auto &block : func.getBlocks()) {
            for (auto &op : block.getOperations()) {
                // op.dump();
                // TODO：use a map to replace if-else
                if (llvm::isa<infinimlir::InputOp>(op)) {
                    handleInputOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::WeightOp>(op)) {
                    handleWeightOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::BatchNormOp>(op)) {
                    handleBatchNormOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::ConvOp>(op)) {
                    handleConvOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::ReluOp>(op)) {
                    handleReluOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::MaxPoolOp>(op)) {
                    handleMaxPoolOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::AddOp>(op)) {
                    handleAddOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::AvgPoolOp>(op)) {
                    handleAvgPoolOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::ReshapeOp>(op)) {
                    handleReshapeOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::MatMulOp>(op)) {
                    handleMatMulOp(g, &op, tensorMap);
                } else if (llvm::isa<infinimlir::ConstantOp>(op)) {
                    handleConstantOp(g, &op, tensorMap);
                } else if (llvm::isa<mlir::func::ReturnOp>(op) || llvm::isa<infinimlir::NoneOp>(op)) {
                    continue;
                } else {
                    throw std::runtime_error("Unsupported op");
                }
            }
        }
    }
    return g;
}

void handleAddOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto addOp = llvm::cast<infinimlir::AddOp>(op);

    // create op inputs Tensor
    std::vector<Tensor> inputs;
    for (unsigned i = 0; i < addOp.getNumOperands(); ++i) {
        mlir::Value value = addOp.getOperand(i);
        if (tensorMap.find(value) == tensorMap.end()) {
            auto shape =
                mlir::cast<mlir::RankedTensorType>(value.getType()).getShape();
            Tensor new_tensor = g->addTensor(
                int64t_to_int(shape),
                convertMlirTypeToDataType(
                    mlir::cast<mlir::RankedTensorType>(value.getType())
                        .getElementType()));
            tensorMap[value] = new_tensor;
            inputs.push_back(new_tensor);
        } else {
            inputs.push_back(tensorMap[value]);
        }
    }

    // create op output Tensor
    mlir::Value output = addOp.getResult();
    auto shape =
        mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor =
        g->addTensor(int64t_to_int(shape),
                     convertMlirTypeToDataType(
                         mlir::cast<mlir::RankedTensorType>(output.getType())
                             .getElementType()));
    tensorMap[output] = output_tensor;
    // create op
    g->addOpWithOutputs<AddObj>(inputs[0], inputs[1], output_tensor);
}

void handleConstantOp(Graph &g, mlir::Operation *op,
                      llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto constantOp = llvm::cast<infinimlir::ConstantOp>(op);

    // create op output Tensor
    mlir::Value output = constantOp.getResult();
    auto shape =
        mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor =
        g->addTensor(int64t_to_int(shape),
                     convertMlirTypeToDataType(
                         mlir::cast<mlir::RankedTensorType>(output.getType())
                             .getElementType()));
    void *data_ptr = (void *)(uintptr_t)constantOp.getDataPtr();
    output_tensor->setDataBlob(make_ref<BlobObj>(g->getRuntime(), data_ptr));
    output_tensor->setWeight();
    tensorMap[output] = output_tensor;
}

void handleInputOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto inputOp = llvm::cast<infinimlir::InputOp>(op);

    // create op inputs Tensor
    std::vector<Tensor> inputs;
   
    mlir::Value value = inputOp.getResult();
    if (tensorMap.find(value) == tensorMap.end()) {
        auto shape =
            mlir::cast<mlir::RankedTensorType>(value.getType()).getShape();
        Tensor new_tensor = g->addTensor(
            int64t_to_int(shape),
            convertMlirTypeToDataType(
                mlir::cast<mlir::RankedTensorType>(value.getType())
                    .getElementType()));
        tensorMap[value] = new_tensor;
        inputs.push_back(new_tensor);
    } else {
        inputs.push_back(tensorMap[value]);
    }
}
void handleWeightOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto weightOp = llvm::cast<infinimlir::WeightOp>(op);

    mlir::Value output = weightOp.getResult();
    auto shape =
        mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor =
        g->addTensor(int64t_to_int(shape),
                     convertMlirTypeToDataType(
                         mlir::cast<mlir::RankedTensorType>(output.getType())
                             .getElementType()));
    auto data = weightOp.read_as_float();
    output_tensor->setDataBlob(make_ref<BlobObj>(g->getRuntime(), data->data()));
    output_tensor->setWeight();
    tensorMap[output] = output_tensor;
}
void handleBatchNormOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto batchNormOp = llvm::cast<infinimlir::BatchNormOp>(op);
    
    // 获取输入tensor
    std::vector<Tensor> inputs;
    for (unsigned i = 0; i < batchNormOp.getNumOperands(); ++i) {
        inputs.push_back(tensorMap[batchNormOp.getOperand(i)]);
    }
    
    // 创建输出tensor
    mlir::Value output = batchNormOp.getResult();
    auto shape = mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor = g->addTensor(
        int64t_to_int(shape),
        convertMlirTypeToDataType(
            mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType()));
    tensorMap[output] = output_tensor;

    // 创建BatchNorm算子
    g->addOpWithOutputs<BatchNormObj>(inputs[0], output_tensor, inputs[1], inputs[2], 
                                     inputs[3], inputs[4]);
}
void handleConvOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto convOp = llvm::cast<infinimlir::ConvOp>(op);
    
    // 获取输入和权重tensor
    Tensor input = tensorMap[convOp.getOperand(0)];
    Tensor weight = tensorMap[convOp.getOperand(1)];
    
    // 创建输出tensor
    mlir::Value output = convOp.getResult();
    auto shape = mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor = g->addTensor(
        int64t_to_int(shape),
        convertMlirTypeToDataType(
            mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType()));
    tensorMap[output] = output_tensor;
    
    // 获取Conv参数
    auto kernelShape = convOp.getKernelShape();
    auto strides = convOp.getStrides();
    auto paddings = convOp.getPads();
    auto dilations = convOp.getDilations();
    
    auto stridesArr = strides.getValue();
    auto paddingsArr = paddings.getValue();
    auto dilationsArr = dilations.has_value() ? 
        dilations.getValue() : 
        mlir::ArrayAttr::get(op->getContext(), {
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 64), 1),
            mlir::IntegerAttr::get(mlir::IntegerType::get(op->getContext(), 64), 1)
        });

    // 创建Conv算子
    g->addOpWithOutputs<ConvObj>(input, weight, output_tensor,
                            stridesArr[0].cast<IntegerAttr>().getInt(),
                            stridesArr[1].cast<IntegerAttr>().getInt(),
                            paddingsArr[0].cast<IntegerAttr>().getInt(),
                            paddingsArr[1].cast<IntegerAttr>().getInt(),
                            dilationsArr[0].cast<IntegerAttr>().getInt(),
                            dilationsArr[1].cast<IntegerAttr>().getInt());
}
void handleReluOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    // TODO: Implement this function
}
void handleMaxPoolOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto poolOp = llvm::cast<infinimlir::MaxPoolOp>(op);
    
    // 获取输入tensor
    Tensor input = tensorMap[poolOp.getOperand(0)];
    
    // 创建输出tensor
    mlir::Value output = poolOp.getResult();
    auto shape = mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor = g->addTensor(
        int64t_to_int(shape),
        convertMlirTypeToDataType(
            mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType()));
    tensorMap[output] = output_tensor;
    
    // 获取Pool参数
    auto kernelShape = poolOp.getKernelShape();
    auto strides = poolOp.getStrides();
    auto paddings = poolOp.getPads();
    
    // 创建MaxPool算子
    g->addOpWithOutputs<MaxPoolObj>(input, output_tensor,
                                   static_cast<int>(kernelShape[0]),
                                   static_cast<int>(kernelShape[1]),
                                   0, 0,
                                   static_cast<int>(paddings[0]),
                                   static_cast<int>(paddings[1]),
                                   static_cast<int>(strides[0]),
                                   static_cast<int>(strides[1]),
                                   poolOp.getCeilMode());
}

void handleAvgPoolOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto poolOp = llvm::cast<infinimlir::AvgPoolOp>(op);
    
    // 获取输入tensor
    Tensor input = tensorMap[poolOp.getOperand(0)];
    
    // 创建输出tensor
    mlir::Value output = poolOp.getResult();
    auto shape = mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor = g->addTensor(
        int64t_to_int(shape),
        convertMlirTypeToDataType(
            mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType()));
    tensorMap[output] = output_tensor;
    
    // 获取Pool参数
    auto kernelShape = poolOp.getKernelShape();
    auto strides = poolOp.getStrides();
    auto paddings = poolOp.getPads();
    
    // 创建AvgPool算子
    g->addOpWithOutputs<AvgPoolObj>(input, output_tensor,
                                   static_cast<int>(kernelShape[0]),
                                   static_cast<int>(kernelShape[1]),
                                   0, 0,
                                   static_cast<int>(paddings[0]),
                                   static_cast<int>(paddings[1]),
                                   static_cast<int>(strides[0]),
                                   static_cast<int>(strides[1]),
                                   false);
}

void handleReshapeOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto reshapeOp = llvm::cast<infinimlir::ReshapeOp>(op);
    
    // 获取输入tensor
    Tensor input = tensorMap[reshapeOp.getOperand(0)];
    
    // 创建输出tensor
    mlir::Value output = reshapeOp.getResult();
    auto shape = mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor = g->addTensor(
        int64t_to_int(shape),
        convertMlirTypeToDataType(
            mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType()));
    tensorMap[output] = output_tensor;
    
    // 创建Reshape算子
    g->addOpWithOutputs<ReshapeObj>(input, output_tensor, int64t_to_int(reshapeOp.getShape()));
}
void handleMatMulOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap) {
    auto matmulOp = llvm::cast<infinimlir::MatMulOp>(op);
    
    // 获取输入tensors
    Tensor input = tensorMap[matmulOp.getOperand(0)];
    Tensor weight = tensorMap[matmulOp.getOperand(1)];
    Tensor bias = tensorMap[matmulOp.getOperand(2)];

    // 创建输出tensor
    mlir::Value output = matmulOp.getResult();
    auto shape = mlir::cast<mlir::RankedTensorType>(output.getType()).getShape();
    Tensor output_tensor = g->addTensor(
        int64t_to_int(shape),
        convertMlirTypeToDataType(
            mlir::cast<mlir::RankedTensorType>(output.getType()).getElementType()));
    tensorMap[output] = output_tensor;
    
    // 创建MatMul算子
    g->addOpWithOutputs<MatmulObj>(input, weight, output_tensor, matmulOp.getLeftTranspose(), matmulOp.getRightTranspose(), bias);
}


} // namespace infinimlir
} // namespace infini
