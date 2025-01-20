#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "InfiniOps.h"

namespace infini {
namespace infinimlir {

using mlir::OperationPass;
using mlir::ModuleOp;
using mlir::func::FuncOp;

std::unique_ptr<OperationPass<ModuleOp>> createShapeInferPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_CLASSES
#include "Passes.h.inc"

} // namespace infinimlir
} // namespace infini