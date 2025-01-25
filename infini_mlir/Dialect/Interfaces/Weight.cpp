#include "Module.h"
#include "InfiniOps.h"

namespace infini {
namespace infinimlir {
using mlir::RankedTensorType;

template <typename T> std::shared_ptr<std::vector<T>> WeightOp::read() {
  auto op = getOperation();
  auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
  auto type = getOutput().getType().cast<RankedTensorType>();
  return module::weightFile(moduleOp).readTensor<T>(module::getName(op).str(), type);
}

std::shared_ptr<std::vector<float>> WeightOp::read_as_float() {
  auto dtype = module::getStorageType(getOutput());
  if (dtype.isF32()) {
    return read<float>();
  } 
  llvm_unreachable("weight data not support read as float now");
  return nullptr;
}

template std::shared_ptr<std::vector<float>> WeightOp::read();

} // namespace infinimlir
} // namespace infini

