#ifndef INFINI_MLIR_MODULE_ATTR
#define INFINI_MLIR_MODULE_ATTR

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

def Module_Platform: I32EnumAttr<"Platform", "module platform",[
      I32EnumAttrCase<"ONNX",    0>
    ]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "infini::infinimlir::module";
}

#endif // INFINI_MLIR_MODULE_ATTR
