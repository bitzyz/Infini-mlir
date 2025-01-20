#include "Module.h"
#include "ShapeInterface.h"
#include "InfiniOps.h"
#include <numeric>

namespace infini {
namespace infinimlir {

void MatMulOp::shape_inference() {
  std::vector<int64_t> in0_shape = module::getShape(getInput());
  int in0_dims = in0_shape.size();
  auto k = in0_shape[in0_dims - 1];
  std::vector<int64_t> in1_shape = module::getShape(getRight());
  int in1_dims = in1_shape.size();
  bool r_transpose = getRightTranspose();
  bool keep_dims_ = getKeepDims();
  int k_idx = in1_dims - (r_transpose ? 1 : 2);
  int n_idx = in1_dims - (r_transpose ? 2 : 1);
  auto n = in1_shape[n_idx];
  std::vector<int64_t> out_shape;
  if (in0_dims > in1_dims) {
    out_shape = in0_shape;
  } else if (in0_dims == in1_dims) {
    out_shape = in0_shape;
    for (int i = out_shape.size() - 3; i >= 0 ; i--) {
      out_shape[i] = std::max(in0_shape[i], in1_shape[i]);
    }
  } else {
    out_shape = in1_shape;
    for (int i = 1; i <= 2; i++) {
      out_shape[out_shape.size() - i] = in0_shape[in0_dims - i];
      if(i > in0_dims){
        out_shape[out_shape.size() - i] = 1;
      }

    }
  }
  if (in1_dims == 1) {
    ASSERT_THIS(in1_shape[0] == k);
    out_shape.pop_back();
  } else if (in1_shape[k_idx] == k) {
      out_shape[out_shape.size() - 1] = n;
  } else if (in1_dims == 2) {
    auto sum = in1_shape[k_idx];
    while (out_shape.size() > 0 && sum % out_shape.back() == 0 && sum != 1) {
      sum = sum / out_shape.back();
      out_shape.pop_back();
    }
    if (sum != 1) {
      dump();
      llvm_unreachable("shape is illegal");
    }
    out_shape.push_back(n);
  } else {
    out_shape[out_shape.size() - 1] = n;
  }
  if (!keep_dims_) {
    int64_t batch_size = std::accumulate(out_shape.begin(), out_shape.end() - 1,
                                         1, std::multiplies<int64_t>());
    out_shape.resize(2);
    out_shape[0] = batch_size;
    out_shape[1] = n;
  }
  module::setShapeOrVerify(getOutput(), out_shape);
}

} // namespace infinimlir
} // namespace infini
