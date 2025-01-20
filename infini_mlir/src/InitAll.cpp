#include "mlir/IR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "InfiniOps.h"
#include "Transforms/include/Passes.h"

namespace infini {
namespace infinimlir {
void registerAllDialects(mlir::DialectRegistry &registry) {
  registry
      .insert<mlir::func::FuncDialect, infini::infinimlir::InfiniDialect>();
}

void registerAllPasses() {
  registerPasses();
}


} // namespace infinimlir
} // namespace infini