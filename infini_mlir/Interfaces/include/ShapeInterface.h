#pragma once

#include "mlir/IR/OpDefinition.h"

namespace infini {
namespace infinimlir {

void common_shape_inference(mlir::Operation *op);

void broadcast_shape_inference(mlir::Operation *op);

} // namespace infinimlir
} // namespace infini

#include "ShapeInterface.h.inc"
