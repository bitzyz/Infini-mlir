#include "Module.h"
#include "ShapeInterface.h"
#include "InfiniOps.h"

namespace infini {
namespace infinimlir {
using mlir::OpBuilder;

void MaxPoolOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto kernel_shape = module::getI64Array(getKernelShape());
  if (kernel_shape->size() == 0) {
    auto num_dim = input_shape.size() - 2;
    ASSERT_THIS(num_dim > 0);
    std::vector<int64_t> vkernel_shape;
    std::vector<int64_t> vstrides(num_dim, 1);
    std::vector<int64_t> vpads(2 * num_dim, 0);
    for (uint32_t i = 2; i < input_shape.size(); i++) {
      vkernel_shape.push_back(input_shape[i]);
    }
    auto builder = OpBuilder(getContext());
    setKernelShapeAttr(builder.getI64ArrayAttr(vkernel_shape));
    setStridesAttr(builder.getI64ArrayAttr(vstrides));
    setPadsAttr(builder.getI64ArrayAttr(vpads));
    kernel_shape = module::getI64Array(getKernelShape());
  }
  ASSERT_THIS(input_shape.size() > 2);
  auto spacial_rank = static_cast<size_t>(input_shape.size() - 2);
  ASSERT_THIS(spacial_rank == getKernelShape().size());
  ASSERT_THIS(getPads().size() == spacial_rank * 2);
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(input_shape[1]);
  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());

  pads = module::getI64Array(getPads());

  for (size_t i = 0; i < spacial_rank; i++) {
    auto out_dim = (input_spacial_shape[i] + pads->at(i) +
                    pads->at(i + spacial_rank) - kernel_shape->at(i)) /
                       strides->at(i) + 1;
    out_shape.push_back(out_dim);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}

} // namespace infinimlir
} // namespace infini
