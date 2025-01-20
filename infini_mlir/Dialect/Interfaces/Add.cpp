#include "ShapeInterface.h"
#include "InfiniOps.h"

namespace infini {
namespace infinimlir {

void AddOp::shape_inference() {
  broadcast_shape_inference(getOperation());
}

} // namespace infinimlir
} // namespace infini
