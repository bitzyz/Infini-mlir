#ifndef INFINI_MLIR_TRAITS
#define INFINI_MLIR_TRAITS

include "mlir/IR/OpBase.td"

class INFINIMLIROpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "infini::infinimlir::trait";
}

def SupportFuseRelu : INFINIMLIROpTrait<"SupportFuseRelu">;

#endif
