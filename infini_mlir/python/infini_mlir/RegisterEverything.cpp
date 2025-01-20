#include "RegisterEverything.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"

PYBIND11_MODULE(_mlirRegisterEverything, m) {
  m.doc() = "INFINI-MLIR Dialects Registration";

  m.def("register_dialects", [](MlirDialectRegistry registry) {
    mlirRegisterAllDialects(registry);
  });
}
