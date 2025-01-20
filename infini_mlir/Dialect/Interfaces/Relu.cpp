#include "ShapeInterface.h"
#include "InfiniOps.h"

namespace infini {
namespace infinimlir {

void ReluOp::shape_inference() { common_shape_inference(getOperation()); }

} // namespace infinimlir
} // namespace infini
