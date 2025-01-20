#ifndef INFINI_MLIR_TRAITS
#define INFINI_MLIR_TRAITS

include "mlir/IR/OpBase.td"

class INFINIMLIROpTrait<string name> : NativeOpTrait<name> {
  let cppNamespace = "infini::infinimlir::trait";
}

def ScalarProducer : INFINIMLIROpTrait<"ScalarProducer">;
def ScalarConsumer : INFINIMLIROpTrait<"ScalarConsumer">;
def SupportConstant: INFINIMLIROpTrait<"SupportConstant">;
def SupportFuseRelu : INFINIMLIROpTrait<"SupportFuseRelu">;

#endif
