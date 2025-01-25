#pragma once

#include "InfiniOps.h"
#include "core/graph.h"
#include "core/runtime.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "operators/element_wise.h"
#include "operators/pooling.h"
#include "operators/reshape.h"
#include "operators/batch_norm.h"
#include "operators/conv.h"
#include "operators/matmul.h"


namespace infini {

namespace infinimlir {
Graph convertMLIRToInfini(mlir::ModuleOp, Runtime runtime);
void handleAddOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleConstantOp(Graph &g, mlir::Operation *op,
                      llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleInputOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleWeightOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleBatchNormOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleConvOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleReluOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleMaxPoolOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleAvgPoolOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleReshapeOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);
void handleMatMulOp(Graph &g, mlir::Operation *op,
                 llvm::DenseMap<mlir::Value, Tensor> &tensorMap);

} // namespace infinimlir

} // namespace infini
