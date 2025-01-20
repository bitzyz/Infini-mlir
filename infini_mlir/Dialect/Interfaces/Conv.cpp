#include "Module.h"
#include "ShapeInterface.h"
#include "InfiniOps.h"

namespace infini {
namespace infinimlir {

void ConvOp::shape_inference() {
  auto input_shape = module::getShape(getInput());
  auto filter_shape = module::getShape(getWeight());
  ASSERT_THIS(input_shape.size() == filter_shape.size());
  ASSERT_THIS(input_shape.size() > 2);
  auto spacial_rank = static_cast<size_t>(input_shape.size() - 2);
  if (spacial_rank != getKernelShape().size()) {
    ASSERT_THIS(module::isUnranked(getOutput()) == false);
    return;
  }
  ASSERT_THIS(getPads().size() == spacial_rank * 2);
  llvm::SmallVector<int64_t> out_shape;
  out_shape.push_back(input_shape[0]);
  out_shape.push_back(filter_shape[0]);
  auto input_spacial_shape = llvm::ArrayRef(&input_shape[2], spacial_rank);
  auto filter_spacial_shape = llvm::ArrayRef(&filter_shape[2], spacial_rank);
  auto pads = module::getI64Array(getPads());
  auto strides = module::getI64Array(getStrides());
  auto dilation = module::getI64Array(getDilations(), spacial_rank, 1);

  for (size_t i = 0; i < spacial_rank; i++) {
    auto out_dim =
        (input_spacial_shape[i] + pads->at(i) + pads->at(i + spacial_rank) -
         dilation->at(i) * (filter_spacial_shape[i] - 1) - 1) /
            strides->at(i) +
        1;
    out_shape.push_back(out_dim);
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}


} // namespace infinimlir
} // namespace infini
