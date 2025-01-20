#include "Module.h"
#include "ShapeInterface.h"
#include "InfiniOps.h"

namespace infini {
namespace infinimlir {

void ReshapeOp::shape_inference() {
  auto in_shape = module::getShape(getInput());
  auto num = module::getNumElements(getInput());
  std::vector<int64_t> out_shape;
  if (getShape().has_value()) {
    auto shape = module::getI64Array(getShape().value());
    int x = -1;
    for (size_t i = 0; i < shape->size(); i++) {
      auto s = shape->at(i);
      if (s > 0) {
        out_shape.push_back(s);
        num /= s;
      } else if (s == 0) {
        out_shape.push_back(in_shape[i]);
        num /= in_shape[i];
      } else if (s == -1) {
        out_shape.push_back(-1);
        x = i;
      } else {
        dump();
        llvm_unreachable("shape is illegal");
      }
    }
    if (x >= 0) {
      out_shape[x] = num;
    }
    module::setShapeOrVerify(getOutput(), out_shape);
  }

  if (!module::isUnranked(getOutput())) {
    auto num_input = module::getNumElements(getInput());
    auto num_output = module::getNumElements(getOutput());
    ASSERT_THIS(num_input == num_output);
  }
}

} // namespace infinimlir
} // namespace infini
