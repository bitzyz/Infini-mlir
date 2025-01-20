#ifndef INFINIMLIR_C_REGISTER_EVERYTHING_H
#define INFINIMLIR_C_REGISTER_EVERYTHING_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_CAPI_EXPORTED void mlirRegisterAllDialects(MlirDialectRegistry registry);

#ifdef __cplusplus
}
#endif

#endif // INFINIMLIR_C_REGISTER_EVERYTHING_H
