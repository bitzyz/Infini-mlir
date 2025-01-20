#include "RegisterEverything.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/MLIRContext.h"
#include "InfiniOps.h"

void mlirRegisterAllDialects(MlirDialectRegistry registry) {
  static_cast<mlir::DialectRegistry *>(registry.ptr)
      ->insert<mlir::func::FuncDialect, infini::infinimlir::InfiniDialect>();
}
