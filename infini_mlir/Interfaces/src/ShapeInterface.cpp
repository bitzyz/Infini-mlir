#include "ShapeInterface.h"
#include "ShapeInterface.cpp.inc"
#include "Module.h"

namespace infini {
namespace infinimlir {

void common_shape_inference(mlir::Operation *op) {
  if (op->getNumResults() != 1) {
    llvm_unreachable("input and output should be only one");
  }
  auto in = op->getOperand(0);
  auto out = op->getResult(0);
  auto in_shape = module::getShape(in);
  module::setShapeOrVerify(out, in_shape);
}

void broadcast_shape_inference(mlir::Operation *op) {
  if (op->getNumResults() != 1) {
    llvm_unreachable("Only supports one output and two inputs.");
  }
  auto lhs_shape = module::getShape(op->getOperand(0));
  auto out_shape = llvm::SmallVector<int64_t>(lhs_shape);
  if (op->getNumOperands() > 1) {
    for (unsigned i = 1; i < op->getNumOperands(); ++i) {
      auto hs_shape = module::getShape(op->getOperand(i));
      auto tmp_shape = llvm::SmallVector<int64_t>();
      for (auto it :
        llvm::zip_longest(llvm::reverse(out_shape), llvm::reverse(hs_shape))) {
        if (std::get<0>(it) && std::get<0>(it) != 1) {
          tmp_shape.push_back(std::get<0>(it).value());
        } else {
          if (std::get<1>(it))
            tmp_shape.push_back(std::get<1>(it).value());
          else
            tmp_shape.push_back(std::get<0>(it).value());
        }
      }
      out_shape = llvm::SmallVector<int64_t>(llvm::reverse(tmp_shape));
    }
  }
  auto out = op->getResult(0);
  module::setShapeOrVerify(out, out_shape);
}

} // namespace infinimlir
} // namespace infini
