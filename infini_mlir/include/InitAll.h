#pragma once

#include "mlir/IR/Dialect.h"

namespace infini {
namespace infinimlir {

void registerAllDialects(mlir::DialectRegistry &registry);
void registerAllPasses();
} // namespace infinimlir
} // namespace infini

