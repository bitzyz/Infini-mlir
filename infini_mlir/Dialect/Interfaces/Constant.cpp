#include "ShapeInterface.h"
#include "InfiniOps.h"
namespace infini
{
namespace infinimlir
{
void ConstantOp::shape_inference() {
  common_shape_inference(getOperation());
}
} // namespace infinimlir
} // namespace infini


