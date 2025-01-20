#include "ShapeInterface.h"
#include "InfiniOps.h"
namespace infini
{
namespace infinimlir
{
void BatchNormOp::shape_inference() {
  common_shape_inference(getOperation());
}
} // namespace infinimlir
} // namespace infini


